import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from red_physics_engine import REDPhysicsEngine


class REDEnv(gym.Env):
    """
    Custom OpenAI Gymnasium Environment for Reverse Electrodialysis (RED)
    operation and optimization within the Blue-Twin AI architecture.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, csv_path="ARA24_Clean_Master_Enhanced.csv", max_steps=365):
        super(REDEnv, self).__init__()

        # Initialize Physics Engine
        self.physics_engine = REDPhysicsEngine(csv_path=csv_path)

        # Load valid river IDs from dataset
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        self.river_ids = df["River ID"].astype(str).tolist()

        self.current_river_id = None
        self.current_step = 0
        self.max_steps = max_steps  # Simulating up to a full year (365 days)

        # Action Space: Continuous control
        # Action 0: Flow rate ratio (river water vs seawater) [0.1, 10.0]
        # Action 1: Extraction factor adjustment [0.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.0], dtype=np.float32),
            high=np.array([10.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space:
        # [Normalized Concentration, Normalized Potential, Temperature State, Day of Year (norm), Theoretical Potential (norm)]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select a river profile for this episode
        self.current_river_id = np.random.choice(self.river_ids)
        self.current_step = 0

        # Initial environmental state (Day 1 of the year)
        obs = self._get_observation(day_of_year=1)
        info = {"river_id": self.current_river_id}

        return obs, info

    def step(self, action):
        self.current_step += 1
        day_of_year = min(self.current_step, 365)

        # Unpack agent actions
        flow_ratio, extraction_factor = action

        # Fetch monthly theoretical potential from physics engine based on day
        base_potential = self.physics_engine.get_data_by_date(self.current_river_id, day_of_year)
        if base_potential is None or np.isnan(base_potential):
            base_potential = 50.0  # Fallback default value if missing

        # Simulate physics calculations
        # Assume standard seawater (~0.6M -> 600 mM) and river water (~0.02M -> 20 mM)
        c_high = 600.0
        c_low = 20.0 * flow_ratio
        c_low = max(c_low, 1.0)  # prevent division by zero or negative

        # Calculate Nernst potential and internal resistance using physics engine
        nernst_e = self.physics_engine.nernst_potential(c_high, c_low)
        resistance = self.physics_engine.internal_resistance(concentration=c_high)

        # Estimate power output based on action efficiency and potential
        power_output = (nernst_e ** 2 / resistance) * extraction_factor * (base_potential / 100.0)
        power_output = max(0.0, power_output)

        # Reward function: Maximize power output while penalizing extreme/inefficient flow ratios
        reward = power_output - 0.1 * (flow_ratio - 1.0) ** 2

        # Check if episode is complete (end of year simulation)
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation(day_of_year)
        info = {
            "power_output": power_output,
            "nernst_potential": nernst_e,
            "internal_resistance": resistance,
            "day": day_of_year
        }

        return obs, float(reward), terminated, truncated, info

    def _get_observation(self, day_of_year):
        # Retrieve base potential data
        pot = self.physics_engine.get_data_by_date(self.current_river_id, day_of_year)
        if pot is None or np.isnan(pot):
            pot = 50.0

        # System state from physics engine [norm_c, norm_phi, norm_T]
        base_state = self.physics_engine.get_system_state(concentration=600.0, potential=pot)

        # Additional normalized variables: [day_of_year_norm, potential_norm]
        day_norm = day_of_year / 365.0
        pot_norm = min(max(pot / 1000.0, 0.0), 1.0)

        obs = np.array([
            base_state[0],
            base_state[1],
            base_state[2],
            day_norm,
            pot_norm
        ], dtype=np.float32)

        return obs

    def render(self):
        print(f"Step: {self.current_step} | River ID: {self.current_river_id}")


if __name__ == "__main__":
    print("--- Testing RED Gym Environment ---")
    env = REDEnv()
    obs, info = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    print(f"Initial Info: {info}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Sample Step Reward: {reward:.4f}")
    print(f"Step Info: {info}")
    print("✅ RED Gym Environment initialized and tested successfully.")
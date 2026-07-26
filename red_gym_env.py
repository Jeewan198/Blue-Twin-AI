import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from red_physics_engine import REDPhysicsEngine


class REDEnv(gym.Env):
    """
    Custom Gymnasium Environment for Reverse Electrodialysis (RED)
    operation and optimisation within the Blue-Twin AI architecture.

    NOTE ON ARCHITECTURE: this environment currently computes reward using
    REDPhysicsEngine's closed-form Nernst/resistance formulas directly, not
    a trained PINN. Per the Literature Review (Section 2.3.4), the intended
    final architecture trains the RL agent against a PINN-constrained
    environment, not the raw analytic physics engine directly. This class
    is a reasonable analytic-physics MVP/testbed to validate the RL loop
    before the PINN is trained and swapped in as the environment's
    dynamics model -- document this explicitly as the current stage of
    Objective 4 in your Implementation chapter, not the final architecture.
    """

    metadata = {"render_modes": ["human"]}

    # Named constants instead of unexplained magic numbers in the reward function.
    BASE_SEAWATER_CONC = 600.0       # mol/m^3 (~mM numerically equivalent), standard reference seawater
    BASE_RIVER_CONC = 20.0           # mol/m^3, standard reference river water baseline
    MIN_RIVER_CONC = 1.0             # floor to avoid log(0) / division-by-zero in Nernst calc
    POTENTIAL_SCALE_DIVISOR = 100.0  # scales MW-range theoretical potential into the reward's rough magnitude
    FLOW_RATIO_PENALTY_WEIGHT = 0.1  # penalises extreme deviation of flow_ratio from the neutral value of 1.0

    def __init__(self, csv_path="ARA24_Clean_Master_Enhanced.csv", max_steps=365):
        super().__init__()

        self.physics_engine = REDPhysicsEngine(csv_path=csv_path)

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        self.river_ids = df["River ID"].astype(str).tolist()

        # Compute the dataset's actual max theoretical potential once, so observation
        # normalisation is calibrated to real data scale rather than an arbitrary guess.
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        month_cols = [f"Theoretical_MW_{m}" for m in months]
        self.max_potential = pd.to_numeric(
            df[month_cols].values.flatten(), errors="coerce"
        )
        self.max_potential = np.nanmax(self.max_potential)
        if not np.isfinite(self.max_potential) or self.max_potential <= 0:
            self.max_potential = 1.0  # defensive fallback, should not trigger on real ARA24 data

        self.current_river_id = None
        self.current_step = 0
        self.max_steps = max_steps

        # Action Space: Continuous control
        # Action 0: Flow rate ratio (river water vs seawater) [0.1, 10.0]
        # Action 1: Extraction factor adjustment [0.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.0], dtype=np.float32),
            high=np.array([10.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space:
        # [0] Normalized dilute-side concentration (c_low / seawater reference)
        # [1] Normalized theoretical potential for the current day (dataset-scaled)
        # [2] Normalized temperature state
        # [3] Normalized day of year
        # [4] Normalized instantaneous Nernst potential -- reflects the electrochemical
        #     driving force actually produced by the agent's own c_low action, distinct
        #     from [1]'s externally-sourced day-ahead theoretical potential forecast.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # Generous upper bound for normalising Nernst potential; real cell-pair values
        # for river/seawater concentration ratios in this project sit well under this.
        self.NERNST_NORM_MAX = 0.5  # volts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # sets up self.np_random for reproducible episodes

        self.current_river_id = self.np_random.choice(self.river_ids)
        self.current_step = 0

        obs = self._get_observation(day_of_year=1, c_low=self.BASE_RIVER_CONC)
        info = {"river_id": self.current_river_id}

        return obs, info

    def step(self, action):
        self.current_step += 1
        day_of_year = min(self.current_step, 365)

        flow_ratio, extraction_factor = action

        base_potential = self.physics_engine.get_data_by_date(self.current_river_id, day_of_year)
        if base_potential is None or np.isnan(base_potential):
            base_potential = 50.0  # fallback default if a river/day lookup fails

        c_high = self.BASE_SEAWATER_CONC
        c_low = max(self.BASE_RIVER_CONC * flow_ratio, self.MIN_RIVER_CONC)

        nernst_e = self.physics_engine.nernst_potential(c_high, c_low)

        # R_lc (Section 2.1) is the dilute (river-side) compartment's resistance --
        # this is the term the agent's flow_ratio action should actually influence,
        # since it dominates internal ohmic loss and is what the whole project is
        # motivated by (see Literature Review, Section 2.1). Using c_high here would
        # make resistance constant regardless of the agent's action.
        resistance = self.physics_engine.internal_resistance(concentration=c_low)

        power_output = (nernst_e ** 2 / resistance) * extraction_factor * (base_potential / self.POTENTIAL_SCALE_DIVISOR)
        power_output = max(0.0, power_output)

        reward = power_output - self.FLOW_RATIO_PENALTY_WEIGHT * (flow_ratio - 1.0) ** 2

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation(day_of_year, c_low=c_low)
        info = {
            "power_output": power_output,
            "nernst_potential": nernst_e,
            "internal_resistance": resistance,
            "day": day_of_year,
        }

        return obs, float(reward), terminated, truncated, info

    def _get_observation(self, day_of_year, c_low):
        pot = self.physics_engine.get_data_by_date(self.current_river_id, day_of_year)
        if pot is None or np.isnan(pot):
            pot = 50.0

        # Normalise against the dataset's actual max potential, not a fixed guess --
        # this keeps every observation dimension genuinely within the declared [0,1]
        # bounds regardless of which river is selected (verified against the Amazon,
        # the dataset's highest-potential river, during testing).
        pot_norm = min(max(pot / self.max_potential, 0.0), 1.0)
        conc_norm = min(max(c_low / self.BASE_SEAWATER_CONC, 0.0), 1.0)
        temp_norm = min(max(self.physics_engine.T / 320.0, 0.0), 1.0)  # ~320K as a generous upper bound
        day_norm = day_of_year / 365.0

        nernst_e = self.physics_engine.nernst_potential(self.BASE_SEAWATER_CONC, c_low)
        nernst_norm = min(max(nernst_e / self.NERNST_NORM_MAX, 0.0), 1.0)

        obs = np.array([
            conc_norm,
            pot_norm,
            temp_norm,
            day_norm,
            nernst_norm,
        ], dtype=np.float32)

        return obs

    def render(self):
        print(f"Step: {self.current_step} | River ID: {self.current_river_id}")


if __name__ == "__main__":
    print("--- Testing RED Gym Environment ---")
    env = REDEnv()
    obs, info = env.reset(seed=42)
    print(f"Initial Observation: {obs} | within [0,1]? {np.all((obs >= 0) & (obs <= 1))}")
    print(f"Initial Info: {info}")

    # Stress-test against the dataset's highest-potential river (Amazon, River ID '1')
    env.current_river_id = "1"
    obs = env._get_observation(day_of_year=180, c_low=20.0)
    print(f"\nAmazon June observation: {obs}")
    print(f"Within declared [0,1] bounds? {np.all((obs >= 0) & (obs <= 1))}")

    # Confirm resistance now actually responds to the agent's flow_ratio action
    env.current_river_id = "1"
    env.current_step = 179
    _, _, _, _, info_low = env.step(np.array([0.1, 0.5], dtype=np.float32))
    env.current_river_id = "1"
    env.current_step = 179
    _, _, _, _, info_high = env.step(np.array([10.0, 0.5], dtype=np.float32))
    print(f"\nResistance at flow_ratio=0.1:  {info_low['internal_resistance']:.4f}")
    print(f"Resistance at flow_ratio=10.0: {info_high['internal_resistance']:.4f}")
    print("(these should now differ -- resistance responds to the agent's action)")

    print("\n✅ RED Gym Environment initialized and tested successfully.")

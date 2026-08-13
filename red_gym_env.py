import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from red_physics_engine import REDPhysicsEngine


class REDEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    BASE_SEAWATER_CONC = 600.0
    BASE_RIVER_CONC = 20.0
    MIN_RIVER_CONC = 1.0
    POTENTIAL_SCALE_DIVISOR = 100.0
    POWER_SCALE = 1.0e4
    FLOW_RATIO_PENALTY_WEIGHT = 0.01

    def __init__(self, csv_path="ARA24_Clean_Master_Enhanced.csv", max_steps=365,
                 normalize_reward_per_river=False, river_id_subset=None):
        super().__init__()

        self.physics_engine = REDPhysicsEngine(csv_path=csv_path)

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        all_river_ids = df["River ID"].astype(str).tolist()

        # river_id_subset restricts which rivers this environment samples from --
        # used to enforce a proper train/test split (e.g. pass the training river
        # IDs here during training, and the held-out test river IDs here during
        # evaluation), so evaluation measures genuine generalisation rather than
        # performance on rivers the agent already trained on.
        if river_id_subset is not None:
            subset = set(str(r) for r in river_id_subset)
            self.river_ids = [r for r in all_river_ids if r in subset]
            missing = subset - set(self.river_ids)
            if missing:
                print(f"Warning: {len(missing)} river IDs in river_id_subset were not "
                      f"found in the dataset and will be ignored: {list(missing)[:5]}...")
            if not self.river_ids:
                raise ValueError("river_id_subset resulted in zero usable rivers -- check the IDs provided.")
        else:
            self.river_ids = all_river_ids

        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        month_cols = [f"Theoretical_MW_{m}" for m in months]
        self.max_potential = pd.to_numeric(df[month_cols].values.flatten(), errors="coerce")
        self.max_potential = np.nanmax(self.max_potential)
        if not np.isfinite(self.max_potential) or self.max_potential <= 0:
            self.max_potential = 1.0

        # Per-river max potential, for optional per-river reward normalisation
        self.normalize_reward_per_river = normalize_reward_per_river
        river_max = pd.to_numeric(df[month_cols].stack(), errors="coerce").groupby(
            df["River ID"].astype(str).repeat(len(month_cols)).values
        ).max()
        self.river_max_potential = river_max.to_dict()

        self.current_river_id = None
        self.current_step = 0
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=np.array([0.1, 0.0], dtype=np.float32),
            high=np.array([10.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.NERNST_NORM_MAX = 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
            base_potential = 50.0

        c_high = self.BASE_SEAWATER_CONC
        c_low = max(self.BASE_RIVER_CONC * flow_ratio, self.MIN_RIVER_CONC)

        nernst_e = self.physics_engine.nernst_potential(c_high, c_low)
        resistance = self.physics_engine.internal_resistance(concentration=c_low)

        power_density = (nernst_e ** 2) / resistance

        if self.normalize_reward_per_river:
            ref = self.river_max_potential.get(self.current_river_id, self.max_potential)
            ref = ref if (ref and np.isfinite(ref) and ref > 0) else self.max_potential
            potential_norm = min(max(base_potential / ref, 0.0), 1.0)
        else:
            potential_norm = min(max(base_potential / self.max_potential, 0.0), 1.0)

        power_output = power_density * self.POWER_SCALE * extraction_factor * potential_norm
        power_output = max(0.0, power_output)

        reward = power_output - self.FLOW_RATIO_PENALTY_WEIGHT * (flow_ratio - 1.0) ** 2

        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation(day_of_year, c_low=c_low)
        info = {
            "power_output": power_output,
            "power_density": power_density,
            "nernst_potential": nernst_e,
            "internal_resistance": resistance,
            "day": day_of_year,
        }
        return obs, float(reward), terminated, truncated, info

    def _get_observation(self, day_of_year, c_low):
        pot = self.physics_engine.get_data_by_date(self.current_river_id, day_of_year)
        if pot is None or np.isnan(pot):
            pot = 50.0
        pot_norm = min(max(pot / self.max_potential, 0.0), 1.0)
        conc_norm = min(max(c_low / self.BASE_SEAWATER_CONC, 0.0), 1.0)
        temp_norm = min(max(self.physics_engine.T / 320.0, 0.0), 1.0)
        day_norm = day_of_year / 365.0
        nernst_e = self.physics_engine.nernst_potential(self.BASE_SEAWATER_CONC, c_low)
        nernst_norm = min(max(nernst_e / self.NERNST_NORM_MAX, 0.0), 1.0)
        return np.array([conc_norm, pot_norm, temp_norm, day_norm, nernst_norm], dtype=np.float32)

    def render(self):
        print(f"Step: {self.current_step} | River ID: {self.current_river_id}")

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import get_linear_fn
from red_gym_env import REDEnv
from river_split import get_train_test_split


class NormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def action(self, action):
        flow_ratio = ((action[0] + 1.0) / 2.0) * (10.0 - 0.1) + 0.1
        extraction_factor = ((action[1] + 1.0) / 2.0) * (1.0 - 0.0) + 0.0
        return np.array([flow_ratio, extraction_factor], dtype=np.float32)


def main():
    print("--- Initializing Blue-Twin AI Training Pipeline (v3: train/test split) ---")

    train_ids, test_ids = get_train_test_split()
    print(f"Train/test split: {len(train_ids)} training rivers, {len(test_ids)} held-out test rivers "
          f"(test rivers are NEVER seen during training).")

    # CHANGE: river_id_subset=train_ids restricts this environment to sample
    # ONLY from the reserved training rivers -- the held-out test_ids are never
    # seen during training, so evaluate_agent.py's results genuinely measure
    # generalisation to unseen rivers, not just performance on the training
    # distribution (see Dr. Bane's feedback on standard evaluation procedure).
    env = REDEnv(csv_path="ARA24_Clean_Master_Enhanced.csv", max_steps=365,
                 normalize_reward_per_river=True, river_id_subset=train_ids)
    env = NormalizedActionWrapper(env)
    env = Monitor(env)

    print("Running Stable-Baselines3 compatibility check on wrapped environment...")
    check_env(env, warn=True)
    print("Environment check passed successfully.")

    # Eval env used by EvalCallback during training is ALSO restricted to the
    # training rivers -- this internal check is for monitoring training
    # progress only, not a substitute for the held-out evaluation done
    # afterward by evaluate_agent.py on test_ids.
    eval_env = REDEnv(csv_path="ARA24_Clean_Master_Enhanced.csv", max_steps=365,
                       normalize_reward_per_river=True, river_id_subset=train_ids)
    eval_env = NormalizedActionWrapper(eval_env)
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model_v3/",
        log_path="./models/eval_logs_v3/",
        eval_freq=2048,
        deterministic=True,
        render=False
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        seed=42,
        verbose=1,
        tensorboard_log="./red_tensorboard/"
    )

    total_timesteps = 150000
    print(f"\nStarting training for {total_timesteps} timesteps with PPO (v3: proper train/test split)...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print("Training complete.")

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "ppo_red_agent_v3_final")
    model.save(model_path)
    print(f"Final model successfully saved to {model_path}.zip")


if __name__ == "__main__":
    main()

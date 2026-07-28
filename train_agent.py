import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from red_gym_env import REDEnv


class NormalizedActionWrapper(gym.ActionWrapper):
    """
    Wraps the REDEnv action space from its natural asymmetric ranges
    ([0.1, 10.0] for flow ratio, [0.0, 1.0] for extraction factor)
    to a symmetric [-1.0, 1.0] range. This improves PPO training stability
    and gradient scaling for the policy's Gaussian distribution.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def action(self, action):
        # Linear rescaling from [-1, 1] agent output to native environment ranges:
        # Flow ratio: [-1, 1] -> [0.1, 10.0]
        flow_ratio = ((action[0] + 1.0) / 2.0) * (10.0 - 0.1) + 0.1
        # Extraction factor: [-1, 1] -> [0.0, 1.0]
        extraction_factor = ((action[1] + 1.0) / 2.0) * (1.0 - 0.0) + 0.0

        return np.array([flow_ratio, extraction_factor], dtype=np.float32)


def main():
    print("--- Initializing Blue-Twin AI Training Pipeline ---")

    # 1. Instantiate environment, wrap with Action Normalizer and Monitor
    env = REDEnv(csv_path="ARA24_Clean_Master_Enhanced.csv", max_steps=365)
    env = NormalizedActionWrapper(env)
    env = Monitor(env)

    # 2. Run check_env on the wrapped environment as recommended to catch API contract breaks early
    print("Running Stable-Baselines3 compatibility check on wrapped environment...")
    check_env(env, warn=True)
    print("Environment check passed successfully.")

    # 3. Setup separate evaluation environment and EvalCallback to save the best model checkpoint
    eval_env = REDEnv(csv_path="ARA24_Clean_Master_Enhanced.csv", max_steps=365)
    eval_env = NormalizedActionWrapper(eval_env)
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./models/eval_logs/",
        eval_freq=2048,
        deterministic=True,
        render=False
    )

    # 4. Define PPO Agent with a fixed seed for reproducible evaluation runs
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=42,  # Ensures repeatable training runs for your Evaluation chapter
        verbose=1,
        tensorboard_log="./red_tensorboard/"
    )

    # 5. Train the agent
    total_timesteps = 73000
    print(f"\nStarting training for {total_timesteps} timesteps with PPO...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print("Training complete.")

    # 6. Save the final model weights
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "ppo_red_agent_final")
    model.save(model_path)
    print(f"Final model successfully saved to {model_path}.zip")


if __name__ == "__main__":
    main()
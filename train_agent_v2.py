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
        flow_ratio = ((action[0] + 1.0) / 2.0) * (10.0 - 0.1) + 0.1
        extraction_factor = ((action[1] + 1.0) / 2.0) * (1.0 - 0.0) + 0.0
        return np.array([flow_ratio, extraction_factor], dtype=np.float32)


def main():
    print("--- Initializing Blue-Twin AI Training Pipeline (v2: improved) ---")

    # CHANGE 1: normalize_reward_per_river=True. Diagnostic test showed that under
    # global-max normalization, most rivers (all but a handful of exceptionally
    # large ones like the Amazon) produced a reward signal near zero -- meaning
    # the agent had almost no gradient signal to learn from on ~99% of the
    # dataset. Per-river normalization brings every river's reward onto a
    # comparable scale, so the agent can learn a genuinely general policy rather
    # than one dominated by a handful of large rivers.
    env = REDEnv(csv_path="ARA24_Clean_Master_Enhanced.csv", max_steps=365,
                 normalize_reward_per_river=True)
    env = NormalizedActionWrapper(env)
    env = Monitor(env)

    print("Running Stable-Baselines3 compatibility check on wrapped environment...")
    check_env(env, warn=True)
    print("Environment check passed successfully.")

    eval_env = REDEnv(csv_path="ARA24_Clean_Master_Enhanced.csv", max_steps=365,
                       normalize_reward_per_river=True)
    eval_env = NormalizedActionWrapper(eval_env)
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model_v2/",
        log_path="./models/eval_logs_v2/",
        eval_freq=2048,
        deterministic=True,
        render=False
    )

    # CHANGE 2: ent_coef > 0. SB3's PPO defaults to ent_coef=0, meaning nothing
    # explicitly discourages the policy's action distribution from collapsing
    # (std shrinking) prematurely. Our first fixed-reward run converged to its
    # best checkpoint at ~12,000 of 73,000 timesteps and regressed afterward --
    # consistent with premature convergence rather than genuine optimum-finding.
    # A small entropy bonus keeps some exploration alive for longer, giving PPO
    # more chances to discover a better policy before settling.
    #
    # CHANGE 3: linear learning-rate decay. Without decay, a constant 3e-4
    # learning rate late in training can keep nudging an already-converged
    # policy around, contributing to the late-training regression observed
    # previously. Decaying it lets early training move fast and late training
    # stabilise rather than drift.
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
        ent_coef=0.005,   # NEW -- small entropy bonus, see CHANGE 2 above
        seed=42,
        verbose=1,
        tensorboard_log="./red_tensorboard/"
    )

    # CHANGE 4: increased budget. Since the agent was still improving in the
    # rollout/ep_rew_mean curve even as eval performance plateaued/regressed,
    # more timesteps combined with the entropy bonus gives it a genuine chance
    # to keep improving rather than cutting off training arbitrarily early.
    total_timesteps = 150000
    print(f"\nStarting training for {total_timesteps} timesteps with PPO (v2 settings)...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print("Training complete.")

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "ppo_red_agent_v2_final")
    model.save(model_path)
    print(f"Final model successfully saved to {model_path}.zip")


if __name__ == "__main__":
    main()

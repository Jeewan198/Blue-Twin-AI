import numpy as np
from stable_baselines3 import PPO
from red_gym_env import REDEnv
from train_agent import NormalizedActionWrapper


def run_evaluation(model_path, csv_path="ARA24_Clean_Master_Enhanced.csv", episodes=10, seed_base=1000):
    """
    Runs deterministic evaluation rollouts for the trained PPO agent and compares
    it against a static baseline over full annual cycles (365 steps), using the
    SAME river/day sequence for both so the comparison is fair.
    """
    model = PPO.load(model_path)

    # The agent's env MUST be wrapped exactly as it was during training -- the
    # trained policy outputs actions in [-1, 1], which only mean the right thing
    # once passed through NormalizedActionWrapper's rescaling back to the real
    # [0.1, 10.0] / [0.0, 1.0] ranges. Evaluating on an unwrapped env would silently
    # misinterpret every action the agent proposes.
    agent_env = NormalizedActionWrapper(REDEnv(csv_path=csv_path))

    # The baseline uses a plain, unwrapped env, since its action is defined
    # directly in real (raw) units: flow_ratio=1.0 (neutral/no adjustment),
    # extraction_factor=0.5 (moderate, fixed extraction).
    baseline_env = REDEnv(csv_path=csv_path)
    STATIC_ACTION = np.array([1.0, 0.5], dtype=np.float32)

    agent_rewards, baseline_rewards = [], []
    agent_power_output, baseline_power_output = [], []

    print(f"Starting evaluation across {episodes} episodes...")
    for ep in range(episodes):
        episode_seed = seed_base + ep  # same seed -> same river/day sequence for both runs

        # 1. Evaluate trained PPO agent
        obs, info = agent_env.reset(seed=episode_seed)
        agent_river = agent_env.unwrapped.current_river_id
        done = False
        ep_agent_reward, ep_agent_power = 0.0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = agent_env.step(action)
            done = terminated or truncated
            ep_agent_reward += reward
            ep_agent_power += info.get("power_output", 0.0)
        agent_rewards.append(ep_agent_reward)
        agent_power_output.append(ep_agent_power)

        # 2. Evaluate static baseline -- SAME seed, so SAME river and day sequence
        obs, info = baseline_env.reset(seed=episode_seed)
        baseline_river = baseline_env.current_river_id
        done = False
        ep_base_reward, ep_base_power = 0.0, 0.0
        while not done:
            obs, reward, terminated, truncated, info = baseline_env.step(STATIC_ACTION)
            done = terminated or truncated
            ep_base_reward += reward
            ep_base_power += info.get("power_output", 0.0)
        baseline_rewards.append(ep_base_reward)
        baseline_power_output.append(ep_base_power)

        match_flag = "OK" if agent_river == baseline_river else "MISMATCH -- check seeding"
        print(f"  Episode {ep+1}: agent_river={agent_river}, baseline_river={baseline_river} [{match_flag}]")

    mean_agent_reward = np.mean(agent_rewards)
    mean_base_reward = np.mean(baseline_rewards)
    mean_agent_power = np.mean(agent_power_output)
    mean_base_power = np.mean(baseline_power_output)

    if mean_base_power != 0:
        improvement_pct = ((mean_agent_power - mean_base_power) / mean_base_power) * 100
    else:
        improvement_pct = float("nan")
        print("WARNING: baseline mean power is exactly 0 -- improvement % is undefined.")

    print("\n--- Evaluation Results ---")
    print(f"Trained Agent Mean Reward:   {mean_agent_reward:.4f}")
    print(f"Static Baseline Mean Reward: {mean_base_reward:.4f}")
    # NOTE: 'power_output' is the reward-function's internal, dimensionless
    # power-density-based term (see red_env.py step()) -- NOT real-world kWh.
    # Do not report this as kWh in your dissertation without deriving a proper
    # unit conversion first.
    print(f"Trained Agent Mean Power (model units): {mean_agent_power:.4f}")
    print(f"Static Baseline Mean Power (model units): {mean_base_power:.4f}")
    print(f"Performance Improvement:     {improvement_pct:+.2f}%")

    return {
        "agent_rewards": agent_rewards,
        "baseline_rewards": baseline_rewards,
        "agent_power_output": agent_power_output,
        "baseline_power_output": baseline_power_output,
        "improvement_pct": improvement_pct,
    }


if __name__ == "__main__":
    checkpoints = {
        "best_model": "./models/best_model/best_model.zip",
        "final_model": "./models/ppo_red_agent_final.zip",
    }

    all_results = {}
    for label, path in checkpoints.items():
        print(f"\n{'='*60}\nEvaluating checkpoint: {label} ({path})\n{'='*60}")
        try:
            all_results[label] = run_evaluation(path, episodes=10)
        except FileNotFoundError:
            print(f"  Skipped -- file not found at {path}")

    if len(all_results) == 2:
        best_power = np.mean(all_results["best_model"]["agent_power_output"])
        final_power = np.mean(all_results["final_model"]["agent_power_output"])
        print(f"\n{'='*60}\nCheckpoint comparison\n{'='*60}")
        print(f"best_model mean power:  {best_power:.4f}")
        print(f"final_model mean power: {final_power:.4f}")
        winner = "best_model" if best_power >= final_power else "final_model"
        print(f"Recommended checkpoint for your Evaluation chapter: {winner}")
    elif len(all_results) == 1:
        print("\nOnly one checkpoint was found -- comparison skipped. "
              "Check the other path if you expected both to exist.")
    else:
        print("\nNeither checkpoint was found. Check your 'models/' directory paths.")

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from red_gym_env import REDEnv
from train_agent import NormalizedActionWrapper


def load_ef_lookup(csv_path="ARA24_Clean_Master_Enhanced.csv"):
    """
    Loads the real, dataset-provided per-river Extraction Factor (EF) column,
    converted from its native 0-100 scale to the 0-1 scale REDEnv's action
    space expects. This replaces an earlier, arbitrarily-chosen fixed baseline
    extraction_factor (0.5) with a defensible, dataset-grounded value specific
    to each river -- see project notes on why this matters: a large majority of
    the previously-reported v2 "improvement" turned out to come from the
    baseline's extraction_factor being arbitrarily low, not from genuine
    flow-management intelligence.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    ef_col = "Extraction Factor (EF)"
    if ef_col not in df.columns:
        raise KeyError(f"Expected column '{ef_col}' not found in {csv_path}. "
                        f"Available columns containing 'Extraction': "
                        f"{[c for c in df.columns if 'Extraction' in c]}")
    df["River ID"] = df["River ID"].astype(str)
    ef_lookup = (df.set_index("River ID")[ef_col] / 100.0).to_dict()
    return ef_lookup


def run_evaluation(model_path, csv_path="ARA24_Clean_Master_Enhanced.csv", episodes=10, seed_base=1000):
    """
    Runs deterministic evaluation rollouts for the trained PPO agent and compares
    it against a static baseline over full annual cycles (365 steps), using the
    SAME river/day sequence for both so the comparison is fair.
    """
    model = PPO.load(model_path)
    ef_lookup = load_ef_lookup(csv_path)
    default_ef = float(np.mean(list(ef_lookup.values())))  # fallback if a river ID is somehow missing

    # The agent's env MUST be wrapped exactly as it was during training -- the
    # trained policy outputs actions in [-1, 1], which only mean the right thing
    # once passed through NormalizedActionWrapper's rescaling back to the real
    # [0.1, 10.0] / [0.0, 1.0] ranges. Evaluating on an unwrapped env would silently
    # misinterpret every action the agent proposes.
    agent_env = NormalizedActionWrapper(REDEnv(csv_path=csv_path))

    # The baseline uses a plain, unwrapped env. flow_ratio=1.0 remains the
    # principled "no adjustment from reference" choice (see project notes).
    # extraction_factor now comes from each river's REAL, dataset-provided
    # Extraction Factor (EF) value, not an arbitrary constant.
    baseline_env = REDEnv(csv_path=csv_path)

    agent_rewards, baseline_rewards = [], []
    agent_power_output, baseline_power_output = [], []
    baseline_ef_used = []

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

        # 2. Evaluate static baseline -- SAME seed, so SAME river and day sequence.
        # extraction_factor is now this specific river's REAL Extraction Factor
        # from the dataset, not an arbitrary constant.
        obs, info = baseline_env.reset(seed=episode_seed)
        baseline_river = baseline_env.current_river_id
        river_ef = ef_lookup.get(baseline_river, default_ef)
        static_action = np.array([1.0, river_ef], dtype=np.float32)
        baseline_ef_used.append(river_ef)
        done = False
        ep_base_reward, ep_base_power = 0.0, 0.0
        while not done:
            obs, reward, terminated, truncated, info = baseline_env.step(static_action)
            done = terminated or truncated
            ep_base_reward += reward
            ep_base_power += info.get("power_output", 0.0)
        baseline_rewards.append(ep_base_reward)
        baseline_power_output.append(ep_base_power)

        match_flag = "OK" if agent_river == baseline_river else "MISMATCH -- check seeding"
        print(f"  Episode {ep+1}: agent_river={agent_river}, baseline_river={baseline_river} "
              f"[{match_flag}], baseline_EF_used={river_ef:.3f}")

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
    print(f"Baseline extraction_factor used (mean across episodes, real per-river EF/100): "
          f"{np.mean(baseline_ef_used):.4f}")
    # NOTE: 'power_output' is the reward-function's internal, dimensionless
    # power-density-based term (see red_gym_env.py step()) -- NOT real-world kWh.
    # Do not report this as kWh in your dissertation without deriving a proper
    # unit conversion first.
    print(f"Trained Agent Mean Power (model units): {mean_agent_power:.4f}")
    print(f"Static Baseline Mean Power (model units): {mean_base_power:.4f}")
    print(f"Performance Improvement:     {improvement_pct:+.2f}%")

    return {
        "agent_rewards": agent_rewards,
        "baseline_ef_used": baseline_ef_used,
        "baseline_rewards": baseline_rewards,
        "agent_power_output": agent_power_output,
        "baseline_power_output": baseline_power_output,
        "improvement_pct": improvement_pct,
    }


if __name__ == "__main__":
    # Original (v1) checkpoints, plus the new v2 experiment (per-river reward
    # normalisation + entropy bonus + LR decay). Any path not found is skipped
    # automatically -- comment out ones you don't have / don't want to re-run.
    checkpoints = {
        "best_model": "./models/best_model/best_model.zip",
        "final_model": "./models/ppo_red_agent_final.zip",
        "best_model_v2": "./models/best_model_v2/best_model.zip",
        "final_model_v2": "./models/ppo_red_agent_v2_final.zip",
    }

    all_results = {}
    for label, path in checkpoints.items():
        print(f"\n{'='*60}\nEvaluating checkpoint: {label} ({path})\n{'='*60}")
        try:
            all_results[label] = run_evaluation(path, episodes=10)
        except FileNotFoundError:
            print(f"  Skipped -- file not found at {path}")

    if len(all_results) >= 2:
        print(f"\n{'='*60}\nCheckpoint comparison\n{'='*60}")
        summary = {
            label: np.mean(r["agent_power_output"])
            for label, r in all_results.items()
        }
        for label, power in sorted(summary.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{label:20s} mean power: {power:.4f}")
        winner = max(summary, key=summary.get)
        print(f"\nRecommended checkpoint for your Evaluation chapter: {winner}")
    elif len(all_results) == 1:
        print("\nOnly one checkpoint was found -- comparison skipped.")
    else:
        print("\nNo checkpoints were found. Check your 'models/' directory paths.")

    if all_results:
        import json
        import os
        os.makedirs("./results", exist_ok=True)
        output_path = "./results/evaluation_results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved evaluation results to {output_path} for use by visualize_results.py")

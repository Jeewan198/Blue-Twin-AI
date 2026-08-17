"""
inspect_agent_actions.py

Sanity-check script: logs the actual flow_ratio and extraction_factor values
a trained agent chooses, day by day, across several held-out test episodes.
This directly answers "is the agent doing something sensible?" rather than
only looking at the aggregate power_output/reward numbers, which can hide
degenerate behaviour (e.g. always picking one extreme action) even when the
final performance number looks good.

Run this after evaluate_agent.py has confirmed a checkpoint's headline
numbers, as a final check before reporting results in the dissertation.
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from red_gym_env import REDEnv
from train_agent import NormalizedActionWrapper
from river_split import get_train_test_split


def inspect_actions(model_path, csv_path="ARA24_Clean_Master_Enhanced.csv",
                     episodes=5, seed_base=2000, normalize_reward_per_river=False):
    model = PPO.load(model_path)
    _, test_ids = get_train_test_split(csv_path)
    env = NormalizedActionWrapper(REDEnv(csv_path=csv_path, river_id_subset=test_ids,
                                          normalize_reward_per_river=normalize_reward_per_river))

    # Also compute each test river's potential_norm (global-scale), to correlate
    # against chosen flow_ratio and check whether the agent adapts appropriately
    # to river size, not just picks a fixed value regardless of context.
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    month_cols = [f"Theoretical_MW_{m}" for m in months]
    global_max_potential = pd.to_numeric(df[month_cols].values.flatten(), errors="coerce")
    global_max_potential = np.nanmax(global_max_potential)
    river_max = df.set_index(df["River ID"].astype(str))[month_cols].apply(
        pd.to_numeric, errors="coerce").max(axis=1)

    all_flow_ratios = []
    all_extraction_factors = []
    per_episode_summary = []

    print(f"Inspecting actual actions chosen across {episodes} held-out test episodes "
          f"(normalize_reward_per_river={normalize_reward_per_river})...\n")
    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        river_id = env.unwrapped.current_river_id
        river_potential_norm = min(max(river_max.get(river_id, 0) / global_max_potential, 0.0), 1.0)
        done = False
        ep_flow_ratios, ep_extraction_factors = [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            real_action = env.action(action)  # apply the same rescaling env.step() would
            flow_ratio, extraction_factor = float(real_action[0]), float(real_action[1])
            ep_flow_ratios.append(flow_ratio)
            ep_extraction_factors.append(extraction_factor)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        all_flow_ratios.extend(ep_flow_ratios)
        all_extraction_factors.extend(ep_extraction_factors)

        summary = {
            "river_id": river_id,
            "potential_norm": river_potential_norm,
            "flow_ratio_mean": np.mean(ep_flow_ratios),
            "flow_ratio_min": np.min(ep_flow_ratios),
            "flow_ratio_max": np.max(ep_flow_ratios),
            "extraction_mean": np.mean(ep_extraction_factors),
        }
        per_episode_summary.append(summary)
        print(f"  River {river_id} (potential_norm={river_potential_norm:.4f}): "
              f"flow_ratio mean={summary['flow_ratio_mean']:.3f} "
              f"(range {summary['flow_ratio_min']:.3f}-{summary['flow_ratio_max']:.3f}) | "
              f"extraction_factor mean={summary['extraction_mean']:.3f}")

    flow_arr = np.array(all_flow_ratios)
    ext_arr = np.array(all_extraction_factors)

    print(f"\n--- Overall action statistics across {len(flow_arr)} agent decisions ---")
    print(f"flow_ratio:        mean={flow_arr.mean():.3f}, std={flow_arr.std():.3f}, "
          f"min={flow_arr.min():.3f}, max={flow_arr.max():.3f}")
    print(f"extraction_factor: mean={ext_arr.mean():.3f}, std={ext_arr.std():.3f}, "
          f"min={ext_arr.min():.3f}, max={ext_arr.max():.3f}")

    # Does chosen flow_ratio actually correlate with river size (potential_norm),
    # as the analytically-derived optimum says it should? A near-zero or negative
    # correlation would indicate the agent isn't adapting flow_ratio to river
    # context, even though it should according to the underlying physics/reward.
    pn_vals = np.array([s["potential_norm"] for s in per_episode_summary])
    fr_vals = np.array([s["flow_ratio_mean"] for s in per_episode_summary])
    if len(pn_vals) > 1 and pn_vals.std() > 0:
        correlation = np.corrcoef(pn_vals, fr_vals)[0, 1]
        print(f"\nCorrelation between river potential_norm and chosen flow_ratio: {correlation:.3f}")
        print("(Analytically, this SHOULD be strongly positive -- larger/higher-potential "
              "rivers should get a higher optimal flow_ratio. Near-zero or negative here "
              "suggests the agent isn't adapting flow_ratio to river context as expected.)")

    warnings = []
    if flow_arr.std() < 0.01:
        warnings.append("flow_ratio has near-zero variance -- agent may be picking a "
                         "single fixed value regardless of state.")
    if ext_arr.std() < 0.01:
        warnings.append("extraction_factor has near-zero variance -- likely EXPECTED, "
                         "not a concern: nothing in the reward function penalises "
                         "extraction_factor, so always maximising it is the mathematically "
                         "correct, rational policy, not a sign of a degenerate agent.")

    if warnings:
        print("\n--- Flags for further investigation ---")
        for w in warnings:
            print(f"  NOTE: {w}")
    else:
        print("\nNo obvious degenerate patterns detected.")

    return {
        "per_episode_summary": per_episode_summary,
        "flow_ratio_mean": float(flow_arr.mean()),
        "flow_ratio_std": float(flow_arr.std()),
        "extraction_mean": float(ext_arr.mean()),
        "extraction_std": float(ext_arr.std()),
        "warnings": warnings,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("final_model_v4 -- fixed observation space (includes river-relative potential)")
    print("=" * 60)
    inspect_actions("./models/ppo_red_agent_v4_final.zip", episodes=129,
                     normalize_reward_per_river=False)

"""
inspect_agent_actions.py

Sanity-check script: logs the actual flow_ratio and extraction_factor values
a trained agent chooses, day by day, across held-out test episodes. This
directly answers "is the agent doing something sensible?" rather than only
looking at the aggregate power_output/reward numbers, which can hide
degenerate behaviour (e.g. always picking one extreme action) even when the
final performance number looks good.

"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from red_gym_env import REDEnv
from train_agent import NormalizedActionWrapper
from river_split import get_train_test_split


def inspect_actions(model_path, csv_path="ARA24_Clean_Master_Enhanced.csv",
                     episodes=None, seed_base=2000, normalize_reward_per_river=False):
    """
    episodes: number of held-out test episodes to inspect. If None (default),
    automatically covers ALL held-out test rivers
    """
    model = PPO.load(model_path)
    _, test_ids = get_train_test_split(csv_path)
    if episodes is None:
        episodes = len(test_ids)
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
    all_river_relative_pot = []
    all_global_pot = []
    all_effective_extraction = []
    all_ef_limits = []
    cap_violations = 0
    per_episode_summary = []
    rivers_seen = set()

    print(f"Inspecting actual actions chosen across {episodes} held-out test episodes "
          f"(normalize_reward_per_river={normalize_reward_per_river})...\n")

    # Iterate through every distinct held-out river explicitly, forcing each
    # one in turn, rather than relying on REDEnv's internal random choice
    # (np_random.choice in reset()) to eventually cover all of them across
    # `episodes` random draws. Random sampling with replacement does not
    # guarantee full coverage: the total decision count (episodes x 365)
    # stays the same whether every distinct river was checked once, or only
    # a subset was checked unevenly -- so this must be enforced explicitly,
    # not inferred from the total count.
    episode_river_ids = test_ids[:episodes]

    for ep, forced_river_id in enumerate(episode_river_ids):
        obs, info = env.reset(seed=seed_base + ep)
        env.unwrapped.current_river_id = forced_river_id
        env.unwrapped.current_step = 0
        obs = env.unwrapped._get_observation(
            day_of_year=1, c_low=env.unwrapped.BASE_RIVER_CONC)

        river_id = env.unwrapped.current_river_id
        rivers_seen.add(river_id)
        river_potential_norm = min(max(river_max.get(river_id, 0) / global_max_potential, 0.0), 1.0)
        done = False
        ep_flow_ratios, ep_extraction_factors = [], []

        while not done:
            # obs[1] = global-scale pot_norm, obs[5] = river-relative pot_norm
            # (see red_gym_env.py _get_observation) -- both captured BEFORE
            # the action is taken, since that's what actually informed it.
            day_global_pot = float(obs[1])
            day_river_relative_pot = float(obs[5]) if len(obs) > 5 else None

            action, _ = model.predict(obs, deterministic=True)
            real_action = env.action(action)  # apply the same rescaling env.step() would
            flow_ratio, extraction_factor = float(real_action[0]), float(real_action[1])
            ep_flow_ratios.append(flow_ratio)
            ep_extraction_factors.append(extraction_factor)
            all_global_pot.append(day_global_pot)
            if day_river_relative_pot is not None:
                all_river_relative_pot.append(day_river_relative_pot)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Verify the ecological EF cap is actually being respected during
            # real evaluation, not just in the isolated unit test -- this is the
            # v5-specific check that matters most.
            if "effective_extraction_factor" in info:
                eff = info["effective_extraction_factor"]
                limit = info["river_ef_limit"]
                all_effective_extraction.append(eff)
                all_ef_limits.append(limit)
                if eff > limit + 1e-6:
                    cap_violations += 1

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

    # Coverage check: confirms whether the episodes actually touched every
    # distinct held-out river.
    n_distinct = len(rivers_seen)
    n_target = len(test_ids)
    if n_distinct < n_target:
        print(f"\nWARNING: only {n_distinct} of {n_target} held-out test rivers were "
              f"actually inspected across these {episodes} episodes.")
    else:
        print(f"\nCoverage check: all {n_distinct} of {n_target} held-out test rivers "
              f"were inspected.")

    flow_arr = np.array(all_flow_ratios)
    ext_arr = np.array(all_extraction_factors)
    correlation_relative = None

    print(f"\n--- Overall action statistics across {len(flow_arr)} agent decisions ---")
    print(f"flow_ratio:        mean={flow_arr.mean():.3f}, std={flow_arr.std():.3f}, "
          f"min={flow_arr.min():.3f}, max={flow_arr.max():.3f}")
    print(f"extraction_factor: mean={ext_arr.mean():.3f}, std={ext_arr.std():.3f}, "
          f"min={ext_arr.min():.3f}, max={ext_arr.max():.3f}")

    # Two separate correlation checks, since v4 trains with normalize_reward_per_river=True
    # -- meaning the reward the agent actually optimised used RIVER-RELATIVE potential,
    # not global. Correlating only against global potential_norm (as done previously)
    # tests the WRONG variable for a per-river-normalised agent.
    pn_vals = np.array([s["potential_norm"] for s in per_episode_summary])
    fr_vals = np.array([s["flow_ratio_mean"] for s in per_episode_summary])
    if len(pn_vals) > 1 and pn_vals.std() > 0:
        correlation_global = np.corrcoef(pn_vals, fr_vals)[0, 1]
        print(f"\n[Per-episode] Correlation between GLOBAL river potential_norm and "
              f"mean flow_ratio: {correlation_global:.3f}")
        print("(This tests whether flow_ratio tracks overall river SIZE -- not "
              "necessarily the right thing to expect if trained with "
              "normalize_reward_per_river=True; see the per-day check below.)")

    if len(all_river_relative_pot) == len(all_flow_ratios) and len(all_river_relative_pot) > 1:
        rr_arr = np.array(all_river_relative_pot)
        if rr_arr.std() > 0:
            correlation_relative = np.corrcoef(rr_arr, flow_arr)[0, 1]
            print(f"\n[Per-DAY] Correlation between RIVER-RELATIVE potential (obs[5]) and "
                  f"flow_ratio, across all {len(rr_arr)} agent decisions: {correlation_relative:.3f}")
            print("(This is the variable the v4 reward actually optimised against -- a "
                  "strong positive value here is the correct test of whether the "
                  "observation-space fix worked, i.e. whether the agent adapts to "
                  "day-to-day, river-relative conditions.)")

    if all_effective_extraction:
        eff_arr = np.array(all_effective_extraction)
        lim_arr = np.array(all_ef_limits)
        print(f"\n--- Ecological extraction constraint check (v5) ---")
        print(f"Effective extraction used:  mean={eff_arr.mean():.4f}, max={eff_arr.max():.4f}")
        print(f"Real river EF limits:       mean={lim_arr.mean():.4f}, max={lim_arr.max():.4f}")
        print(f"Cap violations (effective > real EF limit): {cap_violations} / {len(eff_arr)} decisions")
        if cap_violations == 0:
            print("PASS -- the agent never exceeded any river's real ecological extraction limit.")
        else:
            print("FAIL -- the ecological cap was violated. This should not be possible; "
                  "check red_gym_env.py's step() implementation.")

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
        "correlation_river_relative": float(correlation_relative) if correlation_relative is not None else None,
        "total_decisions": len(eff_arr) if all_effective_extraction else len(flow_arr),
        "rivers_covered": n_distinct,
        "rivers_target": n_target,
        "cap_violations": cap_violations,
        "ecological_compliance_pass": cap_violations == 0,
        "warnings": warnings,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("final_model_v5 -- ecological EF constraint on extraction_factor")
    print("=" * 60)
    # episodes intentionally left unspecified -- inspect_actions now
    # automatically covers all held-out test rivers by default.
    results = inspect_actions("./models/ppo_red_agent_v5_final.zip",
                               normalize_reward_per_river=False)

    import json
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/behavioral_verification.json", "w") as f:
        json.dump({
            "correlation_river_relative": results["correlation_river_relative"],
            "total_decisions": results["total_decisions"],
            "rivers_covered": results["rivers_covered"],
            "rivers_target": results["rivers_target"],
            "cap_violations": results["cap_violations"],
            "ecological_compliance_pass": results["ecological_compliance_pass"],
            "flow_ratio_mean": results["flow_ratio_mean"],
            "flow_ratio_std": results["flow_ratio_std"],
            "extraction_mean": results["extraction_mean"],
        }, f, indent=2)
    print(f"\nSaved behavioral verification results to results/behavioral_verification.json")
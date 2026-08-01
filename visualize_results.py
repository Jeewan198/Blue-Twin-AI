"""
visualize_results.py

Generates dissertation-ready figures from Blue-Twin AI's trained-model
evaluation results. This is deliberately separate from visualize_data.py,
which explores the raw ARA24 dataset and has no connection to the trained
RL agent -- this script is downstream of evaluate_agent.py's output.

Run evaluate_agent.py first; it writes ./results/evaluation_results.json,
which this script reads.
"""
import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt


def find_latest_run(tensorboard_base_dir="./red_tensorboard"):
    """
    Auto-detects the most recent PPO_N run folder, so you don't have to manually
    update the folder number every time you re-run train_agent.py (SB3 increments
    it automatically: PPO_1, PPO_2, PPO_3, ...).
    """
    if not os.path.isdir(tensorboard_base_dir):
        return None

    candidates = glob.glob(os.path.join(tensorboard_base_dir, "PPO_*"))
    numbered = []
    for path in candidates:
        match = re.search(r"PPO_(\d+)$", os.path.basename(path))
        if match:
            numbered.append((int(match.group(1)), path))

    if not numbered:
        return None

    numbered.sort(key=lambda pair: pair[0])
    latest_num, latest_path = numbered[-1]
    print(f"Auto-detected latest training run: {latest_path} (PPO_{latest_num})")
    return latest_path


def load_results(json_path="./results/evaluation_results.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"{json_path} not found. Run evaluate_agent.py first -- it saves "
            f"the results this script needs."
        )
    with open(json_path, "r") as f:
        return json.load(f)


def plot_checkpoint_comparison(results, output_dir):
    """Bar chart: mean power output for each checkpoint vs the static baseline."""
    labels, agent_means, baseline_means = [], [], []
    for label, r in results.items():
        labels.append(label)
        agent_means.append(np.mean(r["agent_power_output"]))
        baseline_means.append(np.mean(r["baseline_power_output"]))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, agent_means, width, label="Trained agent", color="#2E86AB")
    ax.bar(x + width/2, baseline_means, width, label="Static baseline", color="#A23B72")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean power output (model units, see caption)")
    ax.set_title("Trained agent vs static baseline, by checkpoint")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "checkpoint_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def plot_per_episode_power(results, output_dir):
    """Line chart: per-episode power output, agent vs baseline, for each checkpoint."""
    figs = []
    for label, r in results.items():
        agent_power = r["agent_power_output"]
        baseline_power = r["baseline_power_output"]
        episodes = np.arange(1, len(agent_power) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(episodes, agent_power, marker="o", label=f"Trained agent ({label})", color="#2E86AB")
        ax.plot(episodes, baseline_power, marker="s", label="Static baseline", color="#A23B72")
        ax.set_xlabel("Evaluation episode (each a different river, matched seeding)")
        ax.set_ylabel("Total power output over the year (model units)")
        ax.set_title(f"Per-episode power output -- {label}")
        ax.legend()
        ax.set_xticks(episodes)
        fig.tight_layout()

        path = os.path.join(output_dir, f"per_episode_power_{label}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
        figs.append(fig)
    return figs


def plot_improvement_summary(results, output_dir):
    """Bar chart: % improvement over baseline, by checkpoint."""
    labels = list(results.keys())
    improvements = [results[label]["improvement_pct"] for label in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2E86AB" if v >= 0 else "#C1121F" for v in improvements]
    ax.bar(labels, improvements, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Improvement over static baseline (%)")
    ax.set_title("RL agent performance improvement, by checkpoint")
    for i, v in enumerate(improvements):
        ax.text(i, v + (1 if v >= 0 else -3), f"{v:+.2f}%", ha="center")
    fig.tight_layout()

    path = os.path.join(output_dir, "improvement_summary.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    return fig


def plot_training_curves(tensorboard_log_dir, output_dir):
    """
    Reads scalar summaries (ep_rew_mean, explained_variance) from a TensorBoard
    event log and plots them.
    """
    figs = []
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard package not installed -- skipping training curve plots. "
              "Install with: pip install tensorboard")
        return figs

    if tensorboard_log_dir is None or not os.path.isdir(tensorboard_log_dir):
        print(f"TensorBoard log directory not found: {tensorboard_log_dir} -- "
              f"skipping training curve plots.")
        return figs

    ea = EventAccumulator(tensorboard_log_dir)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    wanted = {
        "rollout/ep_rew_mean": "Mean episode reward (training)",
        "train/explained_variance": "Explained variance",
        "eval/mean_reward": "Mean evaluation reward",
    }

    for tag, title in wanted.items():
        if tag not in available_tags:
            print(f"  Tag '{tag}' not found in TensorBoard log -- skipping.")
            continue
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, values, color="#2E86AB")
        ax.set_xlabel("Training timestep")
        ax.set_ylabel(title)
        ax.set_title(title + " over training")
        fig.tight_layout()

        safe_name = tag.replace("/", "_")
        path = os.path.join(output_dir, f"training_{safe_name}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
        figs.append(fig)
    return figs


if __name__ == "__main__":
    output_dir = "./results/figures"
    os.makedirs(output_dir, exist_ok=True)

    results = load_results()

    all_figs = []
    all_figs.append(plot_checkpoint_comparison(results, output_dir))
    all_figs.extend(plot_per_episode_power(results, output_dir))
    all_figs.append(plot_improvement_summary(results, output_dir))

    latest_run = find_latest_run("./red_tensorboard")
    all_figs.extend(plot_training_curves(latest_run, output_dir))

    print(f"\nAll figures saved to {output_dir}")

    print("Opening figures in windows -- close them (or Ctrl+C in the terminal) to finish.")
    try:
        plt.show()
    except Exception as e:
        print(f"Could not open interactive windows ({e}). "
              f"Figures are still saved as PNGs in {output_dir}.")
    finally:
        for fig in all_figs:
            plt.close(fig)

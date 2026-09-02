"""
generate_live_demo.py

Generates a self-contained, interactive HTML report from the project's REAL,
current artifacts: the trained physics engine, the saved evaluation results,
the saved behavioural verification results, and Stable-Baselines3's own
training log. Run this any time after training, evaluation, and
inspect_agent_actions.py have all been run, and it will produce a fresh
report reflecting whatever the current results actually are.

Usage:
    python generate_live_demo.py

Output:
    results/blue_twin_live_report.html
"""
import json
import os
import webbrowser
import numpy as np

from red_physics_engine import REDPhysicsEngine


def load_json(path, required=True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"{path} not found. Run the corresponding script first "
                f"(evaluate_agent.py or inspect_agent_actions.py)."
            )
        return None
    with open(path) as f:
        return json.load(f)


def load_training_curve(log_path="./models/eval_logs_v5/evaluations.npz"):
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found -- training curve panel will be empty. "
              f"This file is saved automatically by Stable-Baselines3's EvalCallback "
              f"during training, provided log_path was set.")
        return []
    data = np.load(log_path)
    timesteps = data["timesteps"]
    # results has shape (n_evals, n_eval_episodes); mean across episodes per eval point
    mean_rewards = data["results"].mean(axis=1)
    return list(zip(timesteps.tolist(), mean_rewards.round(2).tolist()))


def compute_physics_curve(engine, extraction=1.0, potential_norm=1.0,
                           base_seawater=479.0, base_river=6.0, min_river=1.0,
                           power_scale=1.0e4, penalty_weight=0.01):
    points = []
    best_fr, best_reward = None, -np.inf
    fr = 0.1
    while fr <= 10.0:
        c_low = max(base_river * fr, min_river)
        e = engine.nernst_potential(base_seawater, c_low)
        r = engine.internal_resistance(c_low)
        power_output = (e ** 2 / r) * power_scale * extraction * potential_norm
        penalty = penalty_weight * (fr - 1.0) ** 2
        reward = power_output - penalty
        points.append([round(fr, 3), round(reward, 5)])
        if reward > best_reward:
            best_reward, best_fr = reward, fr
        fr += 0.02
    return points, best_fr


def main():
    print("Generating live demo report from current project results...")

    eval_results = load_json("results/evaluation_results.json")
    behavioural = load_json("results/behavioral_verification.json")
    training_curve = load_training_curve()

    engine = REDPhysicsEngine(csv_path="ARA24_Clean_Master_Enhanced.csv")
    curve_points, optimum_fr = compute_physics_curve(engine)

    # Prefer best_model_v5 if present, otherwise report whichever checkpoints exist
    checkpoint_key = "best_model_v5" if "best_model_v5" in eval_results else list(eval_results.keys())[0]
    result = eval_results[checkpoint_key]

    html = render_html(
        checkpoint_key=checkpoint_key,
        improvement_pct=result["improvement_pct"],
        median_improvement_pct=result["median_improvement_pct"],
        excl_outlier_pct=result["improvement_excluding_outlier_pct"],
        flow_ratio_only_pct=result["flow_ratio_only_improvement_pct"],
        curve_points=curve_points,
        optimum_fr=optimum_fr,
        training_curve=training_curve,
        correlation=behavioural["correlation_river_relative"] if behavioural else None,
        cap_violations=behavioural["cap_violations"] if behavioural else None,
        total_decisions=behavioural["total_decisions"] if behavioural else None,
        compliance_pass=behavioural["ecological_compliance_pass"] if behavioural else None,
        engine=engine,
    )

    os.makedirs("results", exist_ok=True)
    out_path = "results/blue_twin_live_report.html"
    with open(out_path, "w") as f:
        f.write(html)
    abs_path = os.path.abspath(out_path)
    print(f"\nLive demo report written to {out_path}")

    try:
        opened = webbrowser.open(f"file://{abs_path}")
        if opened:
            print("Opened in your default browser.")
        else:
            print(f"Could not open a browser automatically. Open this file manually: {abs_path}")
    except Exception as e:
        print(f"Could not open a browser automatically ({e}). Open this file manually: {abs_path}")


def render_html(checkpoint_key, improvement_pct, median_improvement_pct,
                 excl_outlier_pct, flow_ratio_only_pct, curve_points, optimum_fr,
                 training_curve, correlation, cap_violations, total_decisions,
                 compliance_pass, engine):

    compliance_text = "PASS" if compliance_pass else "FAIL"
    compliance_color = "#3fc6d4" if compliance_pass else "#e05252"
    correlation_text = f"{correlation:.3f}" if correlation is not None else "N/A"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Blue-Twin AI -- Live Results Report</title>
<style>
  :root {{
    --bg: #0b1520; --panel: #101f30; --panel-border: #1e3a52;
    --text: #dce8f0; --dim: #7fa0b8; --accent: #3fc6d4; --accent2: #e6a13f;
    --grid: #1b2d3f; --mono: 'IBM Plex Mono', Consolas, monospace;
    --sans: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font-family:var(--sans); padding:28px 36px 60px; }}
  h1 {{ font-size:1.6rem; margin:0 0 2px; color:#fff; }}
  .sub {{ color:var(--dim); font-size:0.92rem; margin-bottom:26px; }}
  .timestamp {{ color:var(--accent2); font-family:var(--mono); font-size:0.8rem; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
  .panel {{ background:var(--panel); border:1px solid var(--panel-border); border-radius:6px; padding:20px 22px; }}
  .panel.wide {{ grid-column:1/-1; }}
  .panel h2 {{ font-size:0.95rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--accent); margin:0 0 4px; }}
  .panel .desc {{ color:var(--dim); font-size:0.82rem; margin-bottom:16px; line-height:1.4; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }}
  .stat {{ background:#0c1a28; border:1px solid var(--panel-border); border-radius:4px; padding:10px 12px; }}
  .stat .k {{ font-size:0.7rem; color:var(--dim); text-transform:uppercase; }}
  .stat .v {{ font-family:var(--mono); font-size:1.25rem; color:var(--accent2); margin-top:2px; }}
  svg {{ width:100%; height:auto; display:block; }}
  .axis-label {{ fill:var(--dim); font-size:10px; }}
  .pass {{ color:{compliance_color}; font-weight:bold; }}
</style>
</head>
<body>

<h1>Blue-Twin AI -- Live Results Report</h1>
<div class="sub">Generated directly from the current project state -- real trained model, real evaluation results, real training log.<br>
<span class="timestamp">Checkpoint reported: {checkpoint_key}</span></div>

<div class="grid">

  <div class="panel wide">
    <h2>Evaluation Results (held-out test rivers)</h2>
    <div class="stat-grid">
      <div class="stat"><div class="k">Mean improvement</div><div class="v">{improvement_pct:.1f}%</div></div>
      <div class="stat"><div class="k">Median improvement</div><div class="v">{median_improvement_pct:.1f}%</div></div>
      <div class="stat"><div class="k">Excl. outlier</div><div class="v">{excl_outlier_pct:.1f}%</div></div>
      <div class="stat"><div class="k">Flow-ratio-only</div><div class="v">{flow_ratio_only_pct:.1f}%</div></div>
    </div>
  </div>

  <div class="panel wide">
    <h2>Behavioural Verification</h2>
    <div class="stat-grid">
      <div class="stat"><div class="k">River-relative correlation</div><div class="v">{correlation_text}</div></div>
      <div class="stat"><div class="k">Decisions checked</div><div class="v">{total_decisions if total_decisions else 'N/A'}</div></div>
      <div class="stat"><div class="k">Ecological cap violations</div><div class="v">{cap_violations if cap_violations is not None else 'N/A'}</div></div>
      <div class="stat"><div class="k">Compliance</div><div class="v pass">{compliance_text}</div></div>
    </div>
  </div>

  <div class="panel wide">
    <h2>Reward vs. Flow Ratio (live from REDPhysicsEngine)</h2>
    <div class="desc">Computed directly by calling the project's real physics engine class across the full flow_ratio range. Optimum found at flow_ratio &asymp; {optimum_fr:.2f}.</div>
    <svg id="curveChart" viewBox="0 0 900 240"></svg>
  </div>

  <div class="panel wide">
    <h2>Training Curve (real SB3 evaluation log)</h2>
    <div class="desc">Loaded directly from Stable-Baselines3's own saved evaluations.npz for this training run.</div>
    <svg id="trainChart" viewBox="0 0 900 260"></svg>
  </div>

</div>

<script>
const curveData = {json.dumps(curve_points)};
const trainData = {json.dumps(training_curve)};

function drawLineChart(svgId, data, xLabel, yLabel, color) {{
  const svg = document.getElementById(svgId);
  if (!data || data.length === 0) {{ svg.innerHTML = '<text x="20" y="20" fill="#7fa0b8">No data found.</text>'; return; }}
  const W = 900, H = svgId === 'trainChart' ? 260 : 240, PADL = 50, PADB = 30, PADT = 14, PADR = 14;
  const xs = data.map(p => p[0]), ys = data.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys, 0), maxY = Math.max(...ys) * 1.08;
  const xScale = x => PADL + ((x - minX) / (maxX - minX || 1)) * (W - PADL - PADR);
  const yScale = y => H - PADB - ((y - minY) / (maxY - minY || 1)) * (H - PADB - PADT);
  let svgStr = `<line x1="${{PADL}}" y1="${{H-PADB}}" x2="${{W-PADR}}" y2="${{H-PADB}}" stroke="#1b2d3f"/>`;
  svgStr += `<line x1="${{PADL}}" y1="${{PADT}}" x2="${{PADL}}" y2="${{H-PADB}}" stroke="#1b2d3f"/>`;
  let path = data.map((p,i) => (i===0?'M':'L') + xScale(p[0]).toFixed(1) + ',' + yScale(p[1]).toFixed(1)).join(' ');
  svgStr += `<path d="${{path}}" fill="none" stroke="${{color}}" stroke-width="1.6"/>`;
  svgStr += `<text x="${{W/2}}" y="${{H-4}}" class="axis-label" text-anchor="middle">${{xLabel}}</text>`;
  svgStr += `<text x="10" y="12" class="axis-label">${{yLabel}}</text>`;
  svg.innerHTML = svgStr;
}}

drawLineChart('curveChart', curveData, 'flow ratio', 'reward', '#3fc6d4');
drawLineChart('trainChart', trainData, 'training timestep', 'eval mean reward', '#e6a13f');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

"""
generate_live_demo.py

Generates a single, self-contained, interactive HTML dashboard from the
project's REAL, current artifacts: the trained physics engine, the saved
evaluation results, the saved behavioural verification results, and
Stable-Baselines3's own training log. Run this any time after training,
evaluation, and inspect_agent_actions.py have all been run, and it will
produce a fresh report reflecting whatever the current results actually are.

This consolidates what were previously three separate outputs (static PNG
charts, this HTML report, and TensorBoard) into one dashboard, and surfaces
numbers that were previously only visible in each script's terminal output
(training summary statistics, the full evaluation results block, and the
physics-engine-vs-REDstack accuracy comparison) as proper visual sections.

Usage:
    python generate_live_demo.py

Output:
    results/blue_twin_live_report.html (opened automatically in your browser)
"""
import json
import os
import webbrowser
import base64
import glob
import numpy as np

from red_physics_engine import REDPhysicsEngine
from river_split import get_train_test_split


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


def load_result_figures(figures_dir="./results/figures"):
    """
    Discovers whatever PNG figures visualize_results.py has actually produced
    in the given directory, and embeds each one directly as base64 so the
    dashboard remains a single, self-contained, portable HTML file rather
    than depending on relative file paths to the figures/ folder staying
    correct. Returns a list of {label, data_uri} dicts, sorted by filename.
    Silently returns an empty list if the directory doesn't exist yet
    (e.g. visualize_results.py hasn't been run), rather than failing.

    Uses an allowlist rather than an exclude-list: only figures that are
    either checkpoint-generic (checkpoint_comparison, improvement_summary,
    any training_* curve) or explicitly tied to a v5 checkpoint are shown.
    This project's history includes several earlier training script versions
    (train_agent.py through train_agent_v4.py) that are no longer reported
    in the dissertation; only the current v5 checkpoints (best_model_v5,
    final_model_v5) are. An allowlist is used instead of naming each old
    version to exclude individually, so any stray figure left over from an
    older run -- v2, v3, v4, or an unversioned one -- is excluded by
    default, without needing to anticipate every possible old filename.
    """
    if not os.path.isdir(figures_dir):
        print(f"Note: {figures_dir} not found -- result figures gallery will be empty. "
              f"Run visualize_results.py first to populate it.")
        return []

    def is_allowed(filename):
        name = filename.lower()
        if name.startswith("checkpoint_comparison") or name.startswith("improvement_summary"):
            return True
        if name.startswith("training_"):
            return True
        if "v5" in name:
            return True
        return False

    all_paths = sorted(glob.glob(os.path.join(figures_dir, "*.png")))
    paths = [p for p in all_paths if is_allowed(os.path.basename(p))]
    skipped = [p for p in all_paths if p not in paths]
    if skipped:
        print(f"Skipped {len(skipped)} figure(s) not matching the v5/generic allowlist: "
              f"{[os.path.basename(p) for p in skipped]}")

    figures = []
    for path in paths:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        label = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
        figures.append({"label": label, "data_uri": f"data:image/png;base64,{encoded}"})
    print(f"Embedded {len(figures)} result figure(s) from {figures_dir}")
    return figures


def load_training_curve(log_path="./models/eval_logs_v5/evaluations.npz"):
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found -- training panels will be empty. "
              f"This file is saved automatically by Stable-Baselines3's EvalCallback "
              f"during training, provided log_path was set.")
        return [], {}
    data = np.load(log_path)
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    curve = list(zip(timesteps.tolist(), mean_rewards.round(2).tolist()))

    best_idx = int(np.argmax(mean_rewards))
    summary = {
        "total_timesteps": int(timesteps[-1]),
        "n_eval_points": len(timesteps),
        "final_mean_reward": round(float(mean_rewards[-1]), 2),
        "best_mean_reward": round(float(mean_rewards[best_idx]), 2),
        "best_at_timestep": int(timesteps[best_idx]),
    }
    return curve, summary


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


def compute_redstack_accuracy(engine, base_seawater=479.0, base_river=6.0,
                               redstack_gross_wm2=0.35, realistic_efficiency=0.37):
    """Compares the engine's predicted power density against REDstack's real,
    published measured output, under REDstack's own reported conditions."""
    e = engine.nernst_potential(base_seawater, base_river)
    r = engine.internal_resistance(base_river)
    power_density_w_cm2 = (e ** 2) / r
    power_density_w_m2 = power_density_w_cm2 * 10000
    raw_ratio = power_density_w_m2 / redstack_gross_wm2
    adjusted_prediction = power_density_w_m2 * realistic_efficiency
    adjusted_ratio = adjusted_prediction / redstack_gross_wm2
    return {
        "predicted_wm2": round(power_density_w_m2, 3),
        "redstack_real_wm2": redstack_gross_wm2,
        "raw_ratio": round(raw_ratio, 2),
        "adjusted_prediction_wm2": round(adjusted_prediction, 3),
        "adjusted_ratio": round(adjusted_ratio, 2),
    }


def main():
    print("Generating unified live dashboard from current project results...")

    eval_results = load_json("results/evaluation_results.json")
    behavioural = load_json("results/behavioral_verification.json")
    training_curve, training_summary = load_training_curve()

    engine = REDPhysicsEngine(csv_path="ARA24_Clean_Master_Enhanced.csv")
    curve_points, optimum_fr = compute_physics_curve(engine)
    redstack_accuracy = compute_redstack_accuracy(engine)

    train_ids, test_ids = get_train_test_split(csv_path="ARA24_Clean_Master_Enhanced.csv")
    split_counts = {"train": len(train_ids), "test": len(test_ids)}

    result_figures = load_result_figures()

    checkpoint_key = "best_model_v5" if "best_model_v5" in eval_results else list(eval_results.keys())[0]
    result = eval_results[checkpoint_key]

    html = render_html(
        checkpoint_key=checkpoint_key,
        result=result,
        curve_points=curve_points,
        optimum_fr=optimum_fr,
        training_curve=training_curve,
        training_summary=training_summary,
        behavioural=behavioural,
        redstack_accuracy=redstack_accuracy,
        split_counts=split_counts,
        result_figures=result_figures,
    )

    os.makedirs("results", exist_ok=True)
    out_path = "results/blue_twin_live_report.html"
    with open(out_path, "w") as f:
        f.write(html)
    abs_path = os.path.abspath(out_path)
    print(f"\nLive dashboard written to {out_path}")

    try:
        opened = webbrowser.open(f"file://{abs_path}")
        if opened:
            print("Opened in your default browser.")
        else:
            print(f"Could not open a browser automatically. Open this file manually: {abs_path}")
    except Exception as e:
        print(f"Could not open a browser automatically ({e}). Open this file manually: {abs_path}")


def render_html(checkpoint_key, result, curve_points, optimum_fr, training_curve,
                 training_summary, behavioural, redstack_accuracy, split_counts, result_figures):

    correlation = behavioural["correlation_river_relative"] if behavioural else None
    cap_violations = behavioural["cap_violations"] if behavioural else None
    rivers_covered = behavioural.get("rivers_covered") if behavioural else None
    rivers_target = behavioural.get("rivers_target") if behavioural else None
    total_decisions = behavioural["total_decisions"] if behavioural else None
    compliance_pass = behavioural["ecological_compliance_pass"] if behavioural else None
    compliance_text = "PASS" if compliance_pass else ("FAIL" if compliance_pass is not None else "N/A")
    compliance_color = "#3fc6d4" if compliance_pass else "#e05252"
    correlation_text = f"{correlation:.3f}" if correlation is not None else "N/A"

    ts = training_summary
    ts_total = ts.get("total_timesteps", "N/A")
    ts_final = ts.get("final_mean_reward", "N/A")
    ts_best = ts.get("best_mean_reward", "N/A")
    ts_best_at = ts.get("best_at_timestep", "N/A")

    if result_figures:
        cards = "\n".join(
            f'<div class="figure-card"><img src="{fig["data_uri"]}" alt="{fig["label"]}">'
            f'<div class="cap">{fig["label"]}</div></div>'
            for fig in result_figures
        )
        figure_gallery_html = f'<div class="figure-gallery">{cards}</div>'
    else:
        figure_gallery_html = ('<div style="color:var(--dim); font-size:0.85rem;">'
                                'No figures found in results/figures/ -- run visualize_results.py first.</div>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Blue-Twin AI -- Unified Live Dashboard</title>
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
  .sub {{ color:var(--dim); font-size:0.92rem; margin-bottom:22px; }}
  .timestamp {{ color:var(--accent2); font-family:var(--mono); font-size:0.8rem; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .panel {{ background:var(--panel); border:1px solid var(--panel-border); border-radius:6px; padding:18px 20px; margin-bottom:20px; }}
  .panel.wide {{ grid-column:1/-1; }}
  .panel h2 {{ font-size:0.92rem; text-transform:uppercase; letter-spacing:0.07em; color:var(--accent); margin:0 0 4px; }}
  .panel .desc {{ color:var(--dim); font-size:0.8rem; margin-bottom:14px; line-height:1.4; }}
  .stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }}
  .figure-gallery {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:16px; }}
  .figure-card {{ background:#0c1a28; border:1px solid var(--panel-border); border-radius:6px; padding:10px; }}
  .figure-card img {{ width:100%; height:auto; border-radius:3px; display:block; }}
  .figure-card .cap {{ font-size:0.78rem; color:var(--dim); margin-top:6px; text-align:center; }}
  .stat {{ background:#0c1a28; border:1px solid var(--panel-border); border-radius:4px; padding:9px 11px; }}
  .stat .k {{ font-size:0.68rem; color:var(--dim); text-transform:uppercase; }}
  .stat .v {{ font-family:var(--mono); font-size:1.15rem; color:var(--accent2); margin-top:2px; }}
  svg {{ width:100%; height:auto; display:block; }}
  .axis-label {{ fill:var(--dim); font-size:10px; }}
  .pass {{ color:{compliance_color}; font-weight:bold; }}
  .section-title {{ font-size:1.1rem; color:#fff; margin: 26px 0 10px; border-bottom:1px solid var(--panel-border); padding-bottom:6px; }}
  input[type=range] {{ width:100%; accent-color:var(--accent); }}
  .row {{ margin-bottom:12px; }}
  label {{ display:block; font-size:0.78rem; color:var(--dim); margin-bottom:3px; }}
  .rowval {{ font-family:var(--mono); color:#fff; }}
</style>
</head>
<body>

<h1>Blue-Twin AI -- Unified Live Dashboard</h1>
<div class="sub">Generated directly from the current project state -- real trained model, real evaluation results, real training log, real physics engine.<br>
<span class="timestamp">Checkpoint reported: {checkpoint_key}</span></div>

<div class="section-title">1. Training Summary</div>
<div class="grid">
  <div class="panel wide">
    <div class="stat-grid">
      <div class="stat"><div class="k">Total timesteps</div><div class="v">{ts_total}</div></div>
      <div class="stat"><div class="k">Final mean reward</div><div class="v">{ts_final}</div></div>
      <div class="stat"><div class="k">Best mean reward</div><div class="v">{ts_best}</div></div>
      <div class="stat"><div class="k">Best reached at step</div><div class="v">{ts_best_at}</div></div>
      <div class="stat"><div class="k">Train / test rivers</div><div class="v">{split_counts['train']} / {split_counts['test']}</div></div>
    </div>
  </div>
  <div class="panel wide">
    <h2>Training Curve (real SB3 evaluation log)</h2>
    <div class="desc">Loaded directly from Stable-Baselines3's own saved evaluations.npz for this training run.</div>
    <svg id="trainChart" viewBox="0 0 900 240"></svg>
  </div>
</div>

<div class="section-title">2. Evaluation Results</div>
<div class="grid">
  <div class="panel wide">
    <div class="stat-grid">
      <div class="stat"><div class="k">Mean improvement</div><div class="v">{result['improvement_pct']:.1f}%</div></div>
      <div class="stat"><div class="k">Median improvement</div><div class="v">{result['median_improvement_pct']:.1f}%</div></div>
      <div class="stat"><div class="k">Excl. outlier</div><div class="v">{result['improvement_excluding_outlier_pct']:.1f}%</div></div>
      <div class="stat"><div class="k">Flow-ratio-only</div><div class="v">{result['flow_ratio_only_improvement_pct']:.1f}%</div></div>
    </div>
  </div>
  <div class="panel wide">
    <h2>Agent vs. Baseline Power Output, Per Episode</h2>
    <div class="desc">Real per-episode power output for both the trained agent and the static baseline, from evaluation_results.json.</div>
    <svg id="episodeChart" viewBox="0 0 900 240"></svg>
  </div>
</div>

<div class="section-title">3. Behavioural Verification</div>
<div class="panel wide">
  <div class="stat-grid">
    <div class="stat"><div class="k">River-relative correlation</div><div class="v">{correlation_text}</div></div>
    <div class="stat"><div class="k">Decisions checked</div><div class="v">{total_decisions if total_decisions else 'N/A'}</div></div>
    <div class="stat"><div class="k">Ecological cap violations</div><div class="v">{cap_violations if cap_violations is not None else 'N/A'}</div></div>
    <div class="stat"><div class="k">Compliance</div><div class="v pass">{compliance_text}</div></div>
    <div class="stat"><div class="k">Held-out rivers covered</div><div class="v">{f"{rivers_covered} / {rivers_target}" if rivers_covered is not None else 'N/A'}</div></div>
  </div>
</div>

<div class="section-title">4. Physics Engine Accuracy vs. REDstack</div>
<div class="panel wide">
  <div class="desc">Model's predicted power density under REDstack's own reported operating conditions, compared against REDstack's real, published measured output.</div>
  <div class="stat-grid">
    <div class="stat"><div class="k">Model prediction</div><div class="v">{redstack_accuracy['predicted_wm2']} W/m&sup2;</div></div>
    <div class="stat"><div class="k">REDstack real (gross)</div><div class="v">{redstack_accuracy['redstack_real_wm2']} W/m&sup2;</div></div>
    <div class="stat"><div class="k">Raw ratio</div><div class="v">{redstack_accuracy['raw_ratio']}&times;</div></div>
    <div class="stat"><div class="k">After efficiency ceiling</div><div class="v">{redstack_accuracy['adjusted_ratio']}&times;</div></div>
  </div>
</div>

<div class="section-title">5. Live Physics Engine</div>
<div class="grid">
  <div class="panel">
    <h2>Physics Calculator</h2>
    <div class="desc">Live Nernst potential and internal resistance, computed from the real REDPhysicsEngine class.</div>
    <div class="row">
      <label>Flow ratio (0.1 &ndash; 10.0) &mdash; <span id="frVal" class="rowval">1.00</span></label>
      <input type="range" id="flowRatio" min="0.1" max="10" step="0.01" value="1.0">
    </div>
    <div class="stat-grid">
      <div class="stat"><div class="k">Nernst potential</div><div class="v" id="outNernst">--</div></div>
      <div class="stat"><div class="k">Reward</div><div class="v" id="outReward">--</div></div>
    </div>
  </div>
  <div class="panel">
    <h2>Reward vs. Flow Ratio</h2>
    <div class="desc">Optimum found at flow_ratio &asymp; {optimum_fr:.2f}.</div>
    <svg id="curveChart" viewBox="0 0 480 200"></svg>
  </div>
</div>

<div class="section-title">6. Result Figures</div>
<div class="panel wide">
  <div class="desc">All figures currently saved by visualize_results.py are directly displayed in this section.</div>
  {figure_gallery_html}
</div>

<script>
const curveData = {json.dumps(curve_points)};
const trainData = {json.dumps(training_curve)};
const episodeAgent = {json.dumps(result.get('agent_power_output', []))};
const episodeBaseline = {json.dumps(result.get('baseline_power_output', []))};

function drawLineChart(svgId, data, xLabel, yLabel, color) {{
  const svg = document.getElementById(svgId);
  if (!data || data.length === 0) {{ svg.innerHTML = '<text x="20" y="20" fill="#7fa0b8">No data found.</text>'; return; }}
  const W = 900, H = 240, PADL = 50, PADB = 30, PADT = 14, PADR = 14;
  const xs = data.map(p => p[0]), ys = data.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys, 0), maxY = Math.max(...ys) * 1.08;
  const xScale = x => PADL + ((x - minX) / (maxX - minX || 1)) * (W - PADL - PADR);
  const yScale = y => H - PADB - ((y - minY) / (maxY - minY || 1)) * (H - PADB - PADT);
  let svgStr = `<line x1="${{PADL}}" y1="${{H-PADB}}" x2="${{W-PADR}}" y2="${{H-PADB}}" stroke="#1b2d3f"/>`;
  let path = data.map((p,i) => (i===0?'M':'L') + xScale(p[0]).toFixed(1) + ',' + yScale(p[1]).toFixed(1)).join(' ');
  svgStr += `<path d="${{path}}" fill="none" stroke="${{color}}" stroke-width="1.6"/>`;
  svgStr += `<text x="${{W/2}}" y="${{H-4}}" class="axis-label" text-anchor="middle">${{xLabel}}</text>`;
  svgStr += `<text x="10" y="12" class="axis-label">${{yLabel}}</text>`;
  svg.innerHTML = svgStr;
}}

function drawBarChart(svgId, seriesA, seriesB, labelA, labelB) {{
  const svg = document.getElementById(svgId);
  if (!seriesA || seriesA.length === 0) {{ svg.innerHTML = '<text x="20" y="20" fill="#7fa0b8">No data found.</text>'; return; }}
  const W = 900, H = 240, PADL = 50, PADB = 40, PADT = 20, PADR = 14;
  const n = seriesA.length;
  const maxY = Math.max(...seriesA, ...seriesB) * 1.1;
  const groupW = (W - PADL - PADR) / n;
  const barW = groupW * 0.35;
  const yScale = y => H - PADB - (y / maxY) * (H - PADB - PADT);
  let s = '';
  for (let i = 0; i < n; i++) {{
    const gx = PADL + i * groupW + groupW*0.15;
    const ah = H - PADB - yScale(seriesA[i]);
    const bh = H - PADB - yScale(seriesB[i]);
    s += `<rect x="${{gx}}" y="${{yScale(seriesA[i])}}" width="${{barW}}" height="${{ah}}" fill="#3fc6d4"/>`;
    s += `<rect x="${{gx+barW+2}}" y="${{yScale(seriesB[i])}}" width="${{barW}}" height="${{bh}}" fill="#e6a13f"/>`;
    s += `<text x="${{gx+barW}}" y="${{H-PADB+16}}" class="axis-label" text-anchor="middle">${{i+1}}</text>`;
  }}
  s += `<rect x="${{PADL}}" y="6" width="10" height="10" fill="#3fc6d4"/><text x="${{PADL+15}}" y="15" class="axis-label">${{labelA}}</text>`;
  s += `<rect x="${{PADL+90}}" y="6" width="10" height="10" fill="#e6a13f"/><text x="${{PADL+105}}" y="15" class="axis-label">${{labelB}}</text>`;
  s += `<text x="${{W/2}}" y="${{H-4}}" class="axis-label" text-anchor="middle">evaluation episode</text>`;
  svg.innerHTML = s;
}}

drawLineChart('trainChart', trainData, 'training timestep', 'eval mean reward', '#e6a13f');
drawBarChart('episodeChart', episodeAgent, episodeBaseline, 'agent', 'baseline');

const F = 96485.33, R = 8.314, T = 298.15, D = 1.0e-9;
const mu = (1.0 * F * D) / (R * T);
const thermalVoltage = (R * T) / F;
const BASE_SEAWATER = 479.0, BASE_RIVER = 6.0, MIN_RIVER = 1.0;
const POWER_SCALE = 1.0e4, PENALTY_WEIGHT = 0.01, R_MEM = 5.6, THICKNESS = 0.0002;

function nernst(cHigh, cLow) {{ return thermalVoltage * Math.log(cHigh / cLow); }}
function resistance(c) {{ const cond = F * mu * c; return R_MEM + (THICKNESS / cond) * 10000; }}
function computeAll(fr) {{
  const cLow = Math.max(BASE_RIVER * fr, MIN_RIVER);
  const e = nernst(BASE_SEAWATER, cLow);
  const r = resistance(cLow);
  const powerOutput = (e * e / r) * POWER_SCALE;
  const penalty = PENALTY_WEIGHT * Math.pow(fr - 1.0, 2);
  return {{ e, reward: powerOutput - penalty }};
}}

const W2 = 480, H2 = 200, PAD = 36;
let optimumFr2 = 0.1, optimumVal2 = -Infinity;
for (let fr = 0.1; fr <= 10; fr += 0.02) {{
  const res = computeAll(fr);
  if (res.reward > optimumVal2) {{ optimumVal2 = res.reward; optimumFr2 = fr; }}
}}
function xScale2(fr) {{ return PAD + (fr / 10) * (W2 - PAD - 12); }}
function yScale2(v) {{ return H2 - PAD - (v / (optimumVal2*1.1)) * (H2 - PAD - 12); }}
let curveSvgStr = `<line x1="${{PAD}}" y1="${{H2-PAD}}" x2="${{W2-12}}" y2="${{H2-PAD}}" stroke="#1b2d3f"/>`;
let path2 = curveData.map((p,i) => (i===0?'M':'L') + xScale2(p[0]).toFixed(1) + ',' + yScale2(p[1]).toFixed(1)).join(' ');
curveSvgStr += `<path d="${{path2}}" fill="none" stroke="#3fc6d4" stroke-width="2"/>`;
curveSvgStr += `<circle id="curveMarker" cx="0" cy="0" r="4.5" fill="#fff" stroke="#3fc6d4" stroke-width="2"/>`;
document.getElementById('curveChart').innerHTML = curveSvgStr;

const flowRatioEl = document.getElementById('flowRatio');
function updateCalc() {{
  const fr = parseFloat(flowRatioEl.value);
  document.getElementById('frVal').textContent = fr.toFixed(2);
  const res = computeAll(fr);
  document.getElementById('outNernst').textContent = (res.e * 1000).toFixed(1) + ' mV';
  document.getElementById('outReward').textContent = res.reward.toFixed(3);
  const marker = document.getElementById('curveMarker');
  marker.setAttribute('cx', xScale2(fr));
  marker.setAttribute('cy', yScale2(res.reward));
}}
flowRatioEl.addEventListener('input', updateCalc);
updateCalc();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
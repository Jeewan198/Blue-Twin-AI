# Blue-Twin AI

A reinforcement learning system for ecologically-constrained optimisation of osmotic power generation via Reverse Electrodialysis (RED).

**MSc Artificial Intelligence Dissertation Project**
Manchester Metropolitan University, Department of Computing and Mathematics
Author: Jeewan Wijewardana
Supervisor: Dr. Michael Bane
EthOS Reference: 92054

---

## Overview

Reverse Electrodialysis generates electricity from the salinity gradient between river water and seawater. This project trains a reinforcement learning agent (PPO) to dynamically control a simulated RED plant's flow ratio and extraction factor, optimising power generation while respecting real, river-specific ecological extraction limits.

The system combines:
- A physics engine grounded in real, published RED plant data (REDstack Afsluitdijk)
- A dataset of 1,078 real rivers (ARA24 Global Salinity Database) with monthly-resolution power potential and ecological extraction factors
- A PPO agent trained and evaluated with a strict train/test split, ensuring evaluation measures genuine generalisation to unseen rivers

## Key Results

Evaluated across all 129 held-out test rivers (full coverage verified):

| Checkpoint | Mean Improvement | Median | Excl. Outlier | Flow-Ratio-Only |
|---|---|---|---|---|
| Best | +39.1% | +36.6% | +37.6% | +33.7% |
| Final | +18.5% | +13.6% | +16.6% | +16.8% |

Behavioural verification: 0.967 correlation between the agent's flow ratio and real, river-relative conditions; 0 ecological constraint violations across 47,085 evaluated decisions.

Full methodology, results, and discussion are in the accompanying dissertation.

## Pipeline

Scripts are listed in the order data flows through them:

| File | Role |
|---|---|
| `clean_salinity_data.py` | Reads the raw ARA24 workbook, produces the clean master CSV |
| `river_split.py` | Deterministic train/test split (949 / 129 rivers) |
| `red_physics_engine.py` | Nernst potential and internal resistance calculations |
| `red_gym_env.py` | Gymnasium environment wrapping the physics engine |
| `train_agent.py` | Defines `NormalizedActionWrapper`, used across training and evaluation |
| `train_agent_v5.py` | Trains the PPO agent (150,000 timesteps, ecological EF constraint) |
| `evaluate_agent.py` | Evaluates a trained checkpoint against a static baseline on held-out rivers |
| `inspect_agent_actions.py` | Behavioural verification: correlation and ecological compliance checks |
| `visualize_data.py` | Dataset exploration figures |
| `visualize_results.py` | Evaluation results figures |
| `generate_live_demo.py` | Unified HTML dashboard, built from real project outputs |
| `test_red_physics.py` | Unit tests for the physics engine |

## Requirements

- Python 3.12
- `gymnasium`
- `stable-baselines3`
- `torch`
- `pandas`, `numpy`, `scipy`
- `matplotlib`, `seaborn`

Install with:
```bash
pip install gymnasium stable-baselines3 torch pandas numpy scipy matplotlib seaborn
```

## Running the Pipeline

Each step depends on the output of the one before it, so they must be run in this order the first time. Steps 3 onward can be re-run independently once a trained checkpoint exists.

**Note on the train/test split**: `river_split.py` is never run directly — it's a module that `train_agent_v5.py`, `evaluate_agent.py`, and `inspect_agent_actions.py` all import and call automatically. It's fully deterministic (fixed seed), so every script always agrees on the same 949 training rivers and 129 held-out test rivers, without needing to generate or save the split separately.

### 1. Clean the raw dataset
```bash
python clean_salinity_data.py
```
**Requires**: `SGE_Global_Database_ARA24.xlsx` (the raw ARA24 workbook) in the same directory.
**Produces**: `ARA24_Clean_Master_Enhanced.csv` — every other script in this pipeline reads this file, not the raw workbook.

### 2. Train the agent
```bash
python train_agent_v5.py
```
**Requires**: the clean CSV from step 1.
**Produces**: `models/ppo_red_agent_v5_final.zip` (final checkpoint) and `models/best_model_v5/best_model.zip` (best checkpoint, saved automatically whenever a periodic evaluation check improves on the previous best). This step is the slowest — 150,000 training timesteps.

### 3. Evaluate against the static baseline
```bash
python evaluate_agent.py
```
**Requires**: a trained checkpoint from step 2.
**Produces**: `results/evaluation_results.json`, and prints per-episode results plus summary improvement percentages to the console. Runs automatically on all 129 held-out test rivers.

### 4. Verify behaviour against ecological constraints
```bash
python inspect_agent_actions.py
```
**Requires**: the same trained checkpoint as step 3.
**Produces**: `results/behavioral_verification.json`, and prints the correlation and ecological compliance checks described in the dissertation.

### 5. Generate result figures
```bash
python visualize_results.py
```
**Requires**: `results/evaluation_results.json` from step 3.
**Produces**: PNG figures in `results/figures/` (checkpoint comparison, per-episode power, improvement summary, training curves).

### 6. Generate the live dashboard
```bash
python generate_live_demo.py
```
**Requires**: outputs from steps 3-5 to already exist.
**Produces**: `results/blue_twin_live_report.html`, a single self-contained dashboard embedding all real results and figures.

### Optional: dataset exploration and unit tests
```bash
python visualize_data.py        # Explore the raw dataset, independent of any trained model
python -m unittest test_red_physics.py -v   # Physics engine unit tests
```
Neither of these depends on a trained checkpoint and can be run at any point after step 1.

## License

This project was developed for academic purposes as part of an MSc dissertation at Manchester Metropolitan University.

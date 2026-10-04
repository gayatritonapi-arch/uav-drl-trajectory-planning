# Deep Reinforcement Learning for Energy-Optimal Trajectory Planning of UAVs

M.Tech project: a DRL agent (SAC/DDPG) learns fuel-efficient climb–cruise–descent
trajectories for a UAV, subject to no-fly zones and wind, benchmarked against a
classical dynamic-programming baseline.

See `docs/Project_Status_Update.docx` for the current status, and
`docs/UAV_DRL_Project_Report.docx` for the full written report.

## Repository Structure

```
.
├── src/                          # All source code
│   ├── uav_trajectory_env.py     # Custom Gymnasium environment (Weeks 1-2)
│   ├── dp_baseline.py            # Classical DP baseline solver (Weeks 3-4)
│   ├── drl_wrappers.py           # Obs scaling / features, reward scaling, wind randomisation
│   ├── train.py                  # SAC / DDPG training with Stable-Baselines3 (Weeks 5-8)
│   └── evaluate.py               # DRL vs DP vs naive comparison + plots
├── data/                         # No static datasets — see data/README.md
├── results/                      # Generated plots / outputs (not hand-edited)
│   ├── sample_trajectory.png
│   └── dp_baseline_trajectory.png
├── docs/                         # Report, slides, and conversation record
│   ├── UAV_DRL_Project_Report.docx
│   ├── UAV_DRL_Trajectory_Planning.pptx
│   ├── Project_Status_Update.docx
│   └── chat_record.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
cd src

# Step 1: sanity-check the environment + naive-policy rollout
python uav_trajectory_env.py

# Step 2: classical DP baseline (same environment, for comparison)
python dp_baseline.py

# Step 3: train DRL agents (~25-30 min each on CPU), then compare
python train.py --algo sac  --timesteps 150000
python train.py --algo ddpg --timesteps 150000
python evaluate.py --runs sac_s0 ddpg_s0
tensorboard --logdir ../results      # optional: live training curves
```

Both scripts save their trajectory plots into `../results/`.

## Project Status

| Weeks | Task | Status |
|---|---|---|
| 1–2 | Environment & dynamics setup | ✅ Done |
| 3–4 | Classical DP baseline | ✅ Done |
| 5–8 | DRL agent training (SAC/DDPG) | ✅ Done |
| 9–10 | Evaluation, Pareto curves, robustness | ⬜ Next step |
| 11–12 | Report writing | ⬜ Planned |

## Key Results (Step 3, environment v1.1)

| Method | Fuel (kg) | vs DP | Flight time | NFZ violations | Decision time |
|---|---|---|---|---|---|
| Naive scripted policy | 1,050 | +39% | 11.3 min | 36 steps (5 km deep) | – |
| Classical DP baseline | 754 | – | 12.2 min | 0 | 29.6 s full re-solve |
| **SAC (primary)** | **527** | **−30%** | 13.7 min | **0** | **0.32 ms** |
| DDPG (secondary) | 599 | −21% | 13.8 min | 0 | 0.23 ms |

Single training seed (seed 0), 150k steps each. SAC trained in ~33 min and DDPG in ~20 min, both on CPU.
Plots: `results/step3_trajectories.png`, `results/step3_learning_curves.png`. Table: `results/step3_comparison.csv`.

**How to read these numbers**
- The DRL agents save fuel mainly by choosing a *slower airspeed* (≈140–200 m/s) at
  low altitude. The DP baseline flies at a fixed 220 m/s, so part of the gap is the
  extra freedom over speed, not only a better route. They trade about 1.5 min of
  flight time for it. That trade-off is what the Weeks 9–10 fuel-vs-time Pareto study
  will measure properly, by sweeping `w_time`.
- The agents did not learn to climb to cruise altitude. The DP baseline does climb.
  This is likely a local optimum, because climbing costs fuel immediately and only
  pays back later. Future work: reward shaping or a speed-aware DP.
- Results come from one seed. Repeat with seeds 1–2 before reporting final numbers.

**Environment v1.1 fixes (found during Step 3):** OpenAP expects airspeed in knots,
but v1.0 passed m/s. That units bug is what caused the earlier "higher altitude
burns more fuel" limitation (report Section 10.2), and it is now resolved. The fuel
model also now uses the actual vertical speed and acceleration, and the altitude
floor is a 500 m minimum safe altitude. Without these fixes, the first SAC run
"reward-hacked" by diving to 0 m at near-idle fuel flow. The v1.0 figures (naive
1,252 kg, DP 487 kg) are superseded by the table above.

## Base Paper

Reference [1] in `docs/UAV_DRL_Project_Report.docx`: *"Fuel- and Noise-Minimal
Departure Trajectory Using Deep Reinforcement Learning with Aircraft Dynamics
and Topography Constraints,"* Transportation Engineering (ScienceDirect), 2025.

## License

Add a license here if required by your institution (e.g., MIT, or leave
unlicensed for academic submission).

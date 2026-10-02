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
│   └── dp_baseline.py            # Classical DP baseline solver (Weeks 3-4)
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
```

Both scripts save their trajectory plots into `../results/`.

## Project Status

| Weeks | Task | Status |
|---|---|---|
| 1–2 | Environment & dynamics setup | ✅ Done |
| 3–4 | Classical DP baseline | ✅ Done |
| 5–8 | DRL agent training (SAC/DDPG) | ⬜ Next step |
| 9–10 | Evaluation, Pareto curves, robustness | ⬜ Planned |
| 11–12 | Report writing | ⬜ Planned |

## Key Results So Far

- **Naive scripted policy:** ~1,252 kg fuel, flies straight through both no-fly zones (demonstrates the problem).
- **Classical DP baseline:** 487 kg fuel, zero no-fly-zone violations (hard constraint).
- **Documented limitation:** both models assume roughly constant airspeed regardless of altitude, which removes the real-world fuel incentive to climb to cruise altitude. This is a shared simplification, so comparisons between the DP baseline and the future DRL agent remain fair. See `docs/UAV_DRL_Project_Report.docx`, Section 10.2, for details.

## Base Paper

Reference [1] in `docs/UAV_DRL_Project_Report.docx`: *"Fuel- and Noise-Minimal
Departure Trajectory Using Deep Reinforcement Learning with Aircraft Dynamics
and Topography Constraints,"* Transportation Engineering (ScienceDirect), 2025.

## License

Add a license here if required by your institution (e.g., MIT, or leave
unlicensed for academic submission).

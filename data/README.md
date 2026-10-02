# Data

This project does not use a static labeled dataset. All data is generated
through simulation at runtime:

- **Aircraft performance data:** sourced from the open-source [OpenAP](https://openap.dev/)
  library (installed as a Python package via `requirements.txt`), which provides
  thrust, drag, and fuel-flow models per aircraft type. No files are stored here —
  OpenAP ships its own reference data as part of the package.
- **Wind fields:** generated synthetically in `src/uav_trajectory_env.py`
  (currently a constant vector; randomize per-episode for the Week 9–10
  robustness evaluation).
- **No-fly zones / airspace layouts:** defined as static circular zones in
  `src/uav_trajectory_env.py`; randomize these per-episode if testing
  generalization across layouts.
- **Training rollouts:** once DRL training starts (Weeks 5–8), the agent
  generates its own state-action-reward data through repeated simulation
  episodes — this is standard for reinforcement learning and does not require
  external data collection.

If you later add real flight-log data (e.g., for validation against actual
aircraft trajectories), place it here and document its source and license in
this file.

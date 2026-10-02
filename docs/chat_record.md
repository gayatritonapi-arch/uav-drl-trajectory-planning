# Chat Record — M.Tech Project: DRL for UAV Energy-Optimal Trajectory Planning

Saved on: 2026-08-30

---

## 1. Initial Brainstorm — Systems Engineering + DRL Project Ideas

Requested M.Tech project ideas combining systems engineering and Deep Reinforcement Learning. Suggestions included:
1. Autonomous resource allocation in cyber-physical systems
2. Predictive maintenance / health management using DRL
3. Multi-agent DRL for traffic/transportation control
4. DRL for UAV/robot swarm coordination
5. Energy management for microgrids using DRL
6. Adaptive control of manufacturing systems
7. DRL-based fault detection & self-healing networks

---

## 2. Narrowed to Avionics Systems

Follow-up ideas specific to avionics:
1. DRL-based adaptive flight control (attitude/altitude hold)
2. Fault-tolerant flight management via DRL
3. DRL for air traffic conflict resolution
4. **Energy-optimal trajectory planning for UAVs/aircraft using DRL** ← chosen topic
5. DRL for avionics health monitoring / prognostics
6. Autonomous emergency landing
7. DRL for cockpit resource management

---

## 3. Scoped to a 3-Month Timeline

Given a 3-month project window, "Energy-Optimal Trajectory Planning for UAVs using DRL" was confirmed feasible if scoped to:
- Single aircraft/UAV
- Point-mass energy dynamics model (not full 6-DOF)
- One phase of flight (climb–cruise–descent)
- Static no-fly zones + simple wind field

### 12-Week Plan
| Weeks | Task |
|---|---|
| 1–2 | Environment, dynamics & fuel model |
| 3–4 | Classical optimal-control baseline |
| 5–8 | SAC/DDPG training & reward tuning |
| 9–10 | Evaluation, Pareto curves, robustness sweeps |
| 11–12 | Report writing, buffer for reruns |

---

## 4. Final Project Title & Description

**Title:**
"Deep Reinforcement Learning for Energy-Optimal Trajectory Planning of UAVs Under Airspace and Environmental Constraints"

**150-word description:**
This project develops a Deep Reinforcement Learning (DRL) based trajectory planning framework for energy-efficient flight profile generation in UAVs/aircraft. Using a point-mass energy dynamics model, a Soft Actor-Critic (SAC) or DDPG agent is trained to generate climb-cruise-descent trajectories that minimize fuel/energy consumption while respecting operational constraints such as static no-fly zones, altitude limits, and simplified wind fields. The agent learns a continuous control policy mapping aircraft state (position, altitude, velocity, remaining fuel) to control actions (thrust, climb rate), balancing the trade-off between energy efficiency, flight time, and safety margins. Performance is benchmarked against a classical optimal control baseline (e.g., dynamic programming) across varying wind and constraint scenarios, with results analyzed via fuel-vs-time Pareto trade-off curves. This work demonstrates the applicability of DRL to real-time, adaptive trajectory optimization problems in aerospace systems engineering, offering a scalable alternative to computationally expensive traditional trajectory optimization methods for future autonomous aerial systems.

---

## 5. MDP Formulation

| Component | Definition |
|---|---|
| **State (9,)** | x, y, altitude, velocity, heading, fuel remaining, distance to goal, wind_x, wind_y |
| **Action (3,)** | thrust command, climb-rate command, heading-rate command (continuous, [-1,1]) |
| **Reward** | `R = −w1·fuel − w2·time − w3·violation − w4·altitude_dev + w5·progress` |
| **Transition** | Point-mass kinematics + OpenAP-based fuel-burn model |

---

## 6. Implementation Roadmap Given

- Custom Gymnasium environment (point-mass dynamics, no-fly zones, wind)
- Classical DP/optimal-control baseline for comparison
- Stable-Baselines3 (SAC primary, DDPG secondary) for training
- Evaluation via fuel-vs-time Pareto curves + robustness sweeps
- Tools: Gymnasium, Stable-Baselines3, OpenAP, NumPy/SciPy, Matplotlib, TensorBoard

---

## 7. PowerPoint Deliverable

An 18-slide PPTX was generated and iteratively refined, covering:
1. Title
2. Motivation ("Why This Problem Matters")
3. Project Objectives
4. MDP Design (state/action/reward)
5. Base Paper Referred
6. Problem Statement
7. Data Source
8. Feasibility Study
9. Tasks Involved / Implementation Steps
10. Reference Papers (5)
11. Expected Outcome
12. System Architecture
13. Tools & Technologies
14. Implementation Timeline (12 weeks)
15. Evaluation Metrics & Expected Outcomes
16. Fuel vs. Time Trade-off Chart (illustrative)
17. Contribution & Significance
18. Thank You

**File delivered:** `UAV_DRL_Trajectory_Planning.pptx`

### Base Paper Referred
"Fuel- and Noise-Minimal Departure Trajectory Using Deep Reinforcement Learning with Aircraft Dynamics and Topography Constraints" — Transportation Engineering (ScienceDirect), 2025.

### 5 Reference Papers
1. Fuel- and noise-minimal departure trajectory using DRL with aircraft dynamics and topography constraints — ScienceDirect, 2025 (Base Paper)
2. Deep Reinforcement Learning for Trajectory Generation and Optimisation of UAVs — DDPG-based quadcopter trajectory optimization, KU Leuven
3. UAV Trajectory Planning in Wireless Sensor Networks for Energy Consumption Minimization by Deep Reinforcement Learning — arXiv:2108.00354
4. Energy-Efficient UAV Trajectory Design for Backscatter Communication: A Deep Reinforcement Learning Approach — IEEE
5. Data Freshness and Energy-Efficient UAV Navigation Optimization: A Deep Reinforcement Learning Approach — IEEE Trans. Intelligent Transportation Systems, 2020

### Data Source
No labeled dataset required — data is simulation-generated:
- **OpenAP** (open-source aircraft performance library: thrust, drag, fuel-flow models)
- Simulated wind fields (synthetic, in-environment)
- Synthetic randomized airspace/no-fly-zone layouts
- Self-generated RL rollouts (state-action-reward trajectories)

### Feasibility Study
- **Technical:** mature open-source tools (Gymnasium, Stable-Baselines3, OpenAP); point-mass model avoids 6-DOF complexity
- **Time (3 months):** single-agent/single-aircraft scope fits 12-week plan; no hardware/flight-test dependency
- **Resource:** runs on a single GPU/CPU workstation; no proprietary datasets; small team sufficient

### Expected Outcome
- A trained SAC/DDPG policy generating fuel-efficient trajectories adapting to wind/no-fly constraints without re-optimization
- Deliverables: trained policy + training curves, baseline comparison results, fuel-vs-time Pareto analysis, final report
- Illustrative targets: ~10–20% fuel savings vs. fixed-profile baseline; real-time inference vs. re-solved optimal control

---

## 8. Category Clarification (CV/NLP/Time Series/Generative AI)

Determined the project doesn't naturally fit Computer Vision, NLP, Time Series Forecasting, or Generative AI — it belongs to **Reinforcement Learning / Intelligent Control Systems**, a distinct ML paradigm. (Later confirmed this categorization wasn't a strict requirement.)

---

## 9. Research Gaps, Objectives, and Novelty

### Identified Gaps (from the 5 reference papers)
1. **Scope limitation in base paper:** optimizes only climb/departure phase, jointly with noise (not fuel-only, not full profile)
2. **Field split:** most "UAV energy DRL" literature actually optimizes communication/networking energy (hovering, relay scheduling), not propulsion/flight fuel
3. **Missing generalization evidence:** no paper explicitly tests policies on out-of-distribution wind/constraints
4. **Missing trade-off characterization:** no Pareto-frontier-style fuel-vs-time analysis in most papers

### Objectives Mapped to Gaps
| Objective | Gap Addressed |
|---|---|
| Optimize full climb–cruise–descent profile for fuel/energy only | Gap 1 |
| Formulate MDP around real flight-propulsion fuel using OpenAP | Gap 2 |
| Evaluate policy on withheld wind/constraint configurations | Gap 3 |
| Produce fuel-vs-time Pareto curve vs. classical DP baseline | Gap 4 |

### Novelty Statement
"This project sits at the intersection of two literatures that don't currently overlap: narrow-scope aerospace fuel-optimal trajectory design (climb-phase only, joint fuel-noise objective) and broad-scope UAV-networking energy DRL (wrong objective — communication, not propulsion). By isolating a full-flight-phase, fuel-only DRL formulation with explicit generalization testing and Pareto-based evaluation, this work fills the specific methodological gap between the two."

---

## 10. Plain-Language Problem Summary

Fixed flight profiles don't adapt to real-time wind/no-fly-zone conditions, wasting fuel. Classical re-optimization is too slow for real-time replanning. Goal: train a DRL "smart pilot" that learns fuel-efficient, adaptive flight paths without recalculating from scratch each time — similar to an experienced driver adjusting routes on the fly vs. a GPS recalculating from zero.

---

## 11. DRL vs. Time Series Forecasting — Clarification

DRL is not a single algorithm; it's a distinct ML paradigm (alongside Supervised and Unsupervised Learning), not a technique within "Time Series Forecasting." Relevant DRL algorithms for this project:
- **DQN** — discrete actions only
- **DDPG** — continuous actions, actor-critic
- **SAC** — continuous actions, more stable/sample-efficient (primary choice)
- **PPO** — general-purpose, discrete or continuous

Forecasting predicts future values passively; RL/DRL actively decides actions to maximize long-term reward.

---

## 12. Step 1 Implementation — Environment & Dynamics (Weeks 1–2)

Built and tested `uav_trajectory_env.py`: a custom Gymnasium-compliant environment implementing the MDP above, using:
- Point-mass kinematics (position, altitude, velocity, heading)
- Real fuel-burn model via **OpenAP** (Airbus A320 profile, swappable)
- Two static circular no-fly zones
- Constant wind vector (to be randomized later for robustness testing)

**Validation:** `check_env()` passed (Gymnasium API compliant). A naive scripted policy (climb then fly straight) reached the goal in 136 steps, burning ~1252 kg fuel, but **flew straight through both no-fly zones** — visually demonstrating exactly the gap a trained DRL agent needs to close.

**Files delivered:**
- `uav_trajectory_env.py`
- `sample_trajectory.png` (top-down path + altitude profile)
- `README.md`
- `requirements.txt`

---

## 13. Step 2 Implementation — Classical DP Baseline (Weeks 3–4)

Built and tested `dp_baseline.py`: a backward dynamic-programming solver over a discretized layered graph (downrange distance × lateral offset × altitude), using the same aircraft/fuel model/goal/no-fly zones as the RL environment.

**Method:** Since downrange distance only increases, the problem is a DAG — solved via backward Bellman recursion, with no-fly-zone violations treated as a **hard constraint** (forbidden transitions) rather than a soft reward penalty.

**Result:** Found a 487 kg fuel path with **zero no-fly-zone violations by construction**.

### Important Documented Finding
The DP baseline chose to stay at low altitude (500 m) for the entire flight rather than climb to the 10,000 m cruise target. Verified this is not a bug — at a fixed true airspeed, OpenAP's fuel model predicts *higher* fuel burn at altitude (0.977 kg/s at 10,000 m vs. 0.665 kg/s at 500 m, both at 220 m/s TAS), which is the opposite of real jet behavior (where higher TAS/Mach at altitude is what makes high-altitude cruise efficient). Since both the RL environment and DP baseline hold airspeed roughly constant regardless of altitude, this is a **shared, documented simplification** — the comparison between DRL and DP remains fair even though neither is fully physically realistic on this point. Flagged as an optional future refinement (altitude-dependent target airspeed).

**Files delivered:**
- `dp_baseline.py`
- `dp_baseline_trajectory.png`
- `README.md` (updated with Step 2 section and the finding above)

---

## 14. Next Step (Not Yet Done)

**Weeks 5–8:** Train the SAC/DDPG agent using Stable-Baselines3 on `uav_trajectory_env.py`, then compare trajectory and fuel usage against the DP baseline from Step 2.

---

## Files Referenced in This Chat (see /mnt/user-data/outputs/)

- `UAV_DRL_Trajectory_Planning.pptx` — 18-slide project presentation
- `uav_drl_step1/uav_trajectory_env.py` — Gymnasium environment
- `uav_drl_step1/dp_baseline.py` — classical DP baseline solver
- `uav_drl_step1/sample_trajectory.png` — naive policy rollout plot
- `uav_drl_step1/dp_baseline_trajectory.png` — DP baseline rollout plot
- `uav_drl_step1/README.md` — full documentation of both steps
- `uav_drl_step1/requirements.txt` — Python dependencies

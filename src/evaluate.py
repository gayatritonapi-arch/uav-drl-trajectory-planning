"""
Step 3 evaluation: trained DRL agents vs. DP baseline vs. naive policy
=======================================================================
Runs every trained agent deterministically on the nominal scenario (same
aircraft, wind, goal and no-fly zones as the DP baseline) and produces:

    ../results/step3_comparison.csv          - metrics table
    ../results/step3_trajectories.png        - top-down + altitude, all methods
    ../results/step3_learning_curves.png     - training + evaluation curves

Usage (from src/):
    python evaluate.py                        # evaluates results/sac_s0, results/ddpg_s0
    python evaluate.py --runs sac_s0 ddpg_s0 sac_windrand_s0
"""

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, DDPG

from uav_trajectory_env import UAVTrajectoryEnv
from drl_wrappers import UAVObsWrapper
from dp_baseline import DPTrajectoryBaseline

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
COLORS = {"Naive": "#9AA0A6", "DP": "#16213E", "SAC": "#F4A825", "DDPG": "#2A9D8F",
          "SAC (wind-rand)": "#E76F51"}


def violation_stats(env, traj_xy):
    steps, max_pen = 0, 0.0
    for p in traj_xy:
        for (cx, cy, r) in env.no_fly_zones:
            d = np.hypot(p[0] - cx, p[1] - cy)
            if d < r:
                steps += 1
                max_pen = max(max_pen, r - d)
    return steps, max_pen


def rollout(policy_fn, wrap=False):
    base = UAVTrajectoryEnv(seed=42)
    env = UAVObsWrapper(base) if wrap else base
    obs, _ = env.reset(seed=0)
    traj = [base._get_obs()[:3].copy()]
    t_inf, n = 0.0, 0
    for _ in range(base.max_steps):
        t = time.perf_counter(); a = policy_fn(obs); t_inf += time.perf_counter() - t; n += 1
        obs, _, term, trunc, info = env.step(a)
        traj.append(base._get_obs()[:3].copy())
        if term or trunc:
            break
    traj = np.array(traj)
    v_steps, v_pen = violation_stats(base, traj[:, :2])
    return dict(traj=traj, fuel=info["fuel_used_total"], steps=n,
                time_min=n * base.dt / 60, reached=bool(info["reached_goal"]),
                viol_steps=v_steps, max_pen_km=v_pen / 1000,
                ms_per_decision=1000 * t_inf / n), base


def naive_policy(obs, cruise=10_000.0):
    climb = 1.0 if obs[2] < cruise - 200 else -0.1
    return np.array([0.3, climb, 0.0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["sac_s0", "ddpg_s0"])
    args = ap.parse_args()

    rows, trajs = [], {}

    # ---- Naive scripted policy ----
    r, env = rollout(naive_policy)
    trajs["Naive"] = r["traj"]
    rows.append(("Naive", r, None))

    # ---- DP baseline ----
    dp = DPTrajectoryBaseline(UAVTrajectoryEnv(seed=42))
    t = time.perf_counter(); path, dp_fuel = dp.solve(); dp_solve = time.perf_counter() - t
    seg = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])).sum()
    trajs["DP"] = path
    rows.append(("DP", dict(fuel=dp_fuel, steps=len(path) - 1, time_min=seg / dp.cruise_speed / 60,
                            reached=True, viol_steps=0, max_pen_km=0.0,
                            ms_per_decision=float("nan")), dp_solve))

    # ---- DRL agents ----
    for run in args.runs:
        d = os.path.join(RESULTS, run)
        algo = SAC if run.startswith("sac") else DDPG
        ckpt = os.path.join(d, "best_model.zip")
        if not os.path.exists(ckpt):
            print(f"skip {run}: no checkpoint"); continue
        model = algo.load(ckpt, device="cpu")
        label = "SAC (wind-rand)" if "windrand" in run else run.split("_")[0].upper()
        r, _ = rollout(lambda o: model.predict(o, deterministic=True)[0], wrap=True)
        trajs[label] = r["traj"]
        rows.append((label, r, None))

    # ---- Metrics table ----
    dp_fuel_ref = rows[1][1]["fuel"]
    out_csv = os.path.join(RESULTS, "step3_comparison.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "fuel_kg", "fuel_vs_DP_pct", "flight_time_min", "decisions",
                    "reached_goal", "nfz_violation_steps", "max_nfz_penetration_km",
                    "ms_per_decision", "dp_solve_s"])
        for name, r, solve in rows:
            w.writerow([name, f"{r['fuel']:.1f}", f"{100 * (r['fuel'] - dp_fuel_ref) / dp_fuel_ref:+.1f}",
                        f"{r['time_min']:.1f}", r["steps"], r["reached"], r["viol_steps"],
                        f"{r['max_pen_km']:.2f}", f"{r['ms_per_decision']:.3f}",
                        f"{solve:.1f}" if solve else ""])
    print(open(out_csv).read())

    # ---- Trajectory plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for (cx, cy, rad) in env.no_fly_zones:
        axes[0].add_patch(plt.Circle((cx / 1000, cy / 1000), rad / 1000, color="red", alpha=0.18))
    for name, tr in trajs.items():
        ls = "--" if name == "Naive" else "-"
        axes[0].plot(tr[:, 0] / 1000, tr[:, 1] / 1000, ls, color=COLORS.get(name), lw=2, label=name)
        axes[1].plot(tr[:, 0] / 1000, tr[:, 2], ls, color=COLORS.get(name), lw=2, label=name)
    axes[0].scatter(*(env.goal_pos / 1000), color="green", marker="*", s=220, zorder=5)
    axes[0].scatter(0, 0, color="blue", s=70, zorder=5)
    axes[0].set(xlabel="x (km)", ylabel="y (km)", title="Top-down trajectories (red = no-fly zones)")
    axes[0].axis("equal"); axes[0].legend()
    axes[1].axhline(env.cruise_alt, color="gray", ls=":", lw=1)
    axes[1].set(xlabel="x (km)", ylabel="Altitude (m)", title="Altitude profiles")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "step3_trajectories.png"), dpi=150)

    # ---- Learning curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for run in args.runs:
        d = os.path.join(RESULTS, run)
        label = "SAC (wind-rand)" if "windrand" in run else run.split("_")[0].upper()
        mon = os.path.join(d, "train_monitor.csv")
        if os.path.exists(mon):
            df = pd.read_csv(mon, skiprows=1)
            steps = df["l"].cumsum()
            axes[0].plot(steps, df["r"].rolling(20, min_periods=1).mean(), color=COLORS.get(label), label=label)
            axes[1].plot(steps, df["fuel_used_total"].rolling(20, min_periods=1).mean(),
                         color=COLORS.get(label), label=label)
        ev = os.path.join(d, "evaluations.npz")
        if os.path.exists(ev):
            e = np.load(ev)
            axes[0].scatter(e["timesteps"], e["results"].mean(1), s=10, color=COLORS.get(label), alpha=0.6)
    axes[1].axhline(dp_fuel_ref, color=COLORS["DP"], ls="--", label=f"DP baseline ({dp_fuel_ref:.0f} kg)")
    axes[0].set(xlabel="Environment steps", ylabel="Episode return (unscaled)",
                title="Training return (line: 20-ep avg, dots: deterministic eval)")
    axes[1].set(xlabel="Environment steps", ylabel="Fuel per episode (kg)", title="Training fuel use (20-ep avg)")
    axes[0].legend(); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "step3_learning_curves.png"), dpi=150)
    print("Saved step3_trajectories.png and step3_learning_curves.png")


if __name__ == "__main__":
    main()

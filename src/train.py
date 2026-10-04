"""
Step 3: DRL Agent Training -- SAC / DDPG (Weeks 5-8)
=====================================================
Trains a Stable-Baselines3 SAC (primary) or DDPG (secondary) agent on the
same `UAVTrajectoryEnv` used by the DP baseline, so results are directly
comparable.

Usage (from src/):
    python train.py --algo sac  --timesteps 150000
    python train.py --algo ddpg --timesteps 150000
    python train.py --algo sac  --timesteps 150000 --randomize-wind   # robustness variant

Outputs (under ../results/<run_name>/):
    best_model.zip      - checkpoint with the best evaluation reward
    final_model.zip     - model at the end of training
    train_monitor.csv   - per-episode reward / length / fuel (training)
    evaluations.npz     - periodic deterministic evaluation results
    tb/                 - TensorBoard logs  (tensorboard --logdir ../results)
    config.json         - every hyperparameter used, for the report
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import SAC, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from drl_wrappers import make_env

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# Hyperparameters chosen for a ~130-step episode, 3-D continuous action task.
HPARAMS = {
    "sac": dict(
        learning_rate=3e-4, buffer_size=300_000, learning_starts=5_000,
        batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
        ent_coef="auto", policy_kwargs=dict(net_arch=[256, 256]),
    ),
    "ddpg": dict(
        learning_rate=1e-4, buffer_size=300_000, learning_starts=5_000,
        batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["sac", "ddpg"], default="sac")
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--randomize-wind", action="store_true")
    p.add_argument("--eval-freq", type=int, default=5_000)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    run = args.run_name or f"{args.algo}{'_windrand' if args.randomize_wind else ''}_s{args.seed}"
    out = os.path.join(RESULTS, run)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    train_env = DummyVecEnv([make_env(args.seed, args.reward_scale, args.randomize_wind,
                                      log_file=os.path.join(out, "train_monitor.csv"))])
    # Evaluation always uses the nominal (DP-baseline) scenario, fixed wind
    eval_env = DummyVecEnv([make_env(args.seed + 1000, args.reward_scale, False)])

    hp = dict(HPARAMS[args.algo])
    if args.algo == "sac":
        model = SAC("MlpPolicy", train_env, seed=args.seed, verbose=0,
                    tensorboard_log=os.path.join(out, "tb"), **hp)
    else:
        n_act = train_env.action_space.shape[0]
        noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_act), sigma=0.3 * np.ones(n_act))
        model = DDPG("MlpPolicy", train_env, action_noise=noise, seed=args.seed, verbose=0,
                     tensorboard_log=os.path.join(out, "tb"), **hp)
        hp["action_noise"] = "OrnsteinUhlenbeck(sigma=0.3)"

    eval_cb = EvalCallback(eval_env, best_model_save_path=out, log_path=out,
                           eval_freq=args.eval_freq, n_eval_episodes=1, deterministic=True)

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "hparams": hp}, f, indent=2, default=str)

    t0 = time.time()
    print(f"[{run}] training {args.algo.upper()} for {args.timesteps:,} steps ...", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=False)
    model.save(os.path.join(out, "final_model"))
    mins = (time.time() - t0) / 60
    print(f"[{run}] done in {mins:.1f} min. Best eval reward: {eval_cb.best_mean_reward:.2f}", flush=True)
    with open(os.path.join(out, "config.json"), "r+") as f:
        cfg = json.load(f); cfg["train_minutes"] = round(mins, 1)
        f.seek(0); json.dump(cfg, f, indent=2, default=str); f.truncate()


if __name__ == "__main__":
    main()

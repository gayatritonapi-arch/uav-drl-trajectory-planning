"""
Step 3 helper: observation / reward wrappers for DRL training (Weeks 5-8)
=========================================================================
The raw `UAVTrajectoryEnv` observation mixes values on wildly different
scales (positions up to 1e5 m, heading in rad, fuel in kg). Neural-network
policies (SAC/DDPG) train poorly on unscaled inputs, so we wrap the env
WITHOUT modifying its physics or reward definition:

  UAVObsWrapper      -> scaled + engineered features (19-dim)
  RewardScaleWrapper -> multiplies reward by a constant (default 0.1)
  WindRandomizationWrapper (optional) -> samples a new wind vector per
                        episode, used for robustness training/testing.

Engineered features (all roughly in [-1, 1]):
   0  x / goal_distance
   1  y / 30 km
   2  altitude / 15 km
   3  (airspeed - 100) / 150
   4  sin(heading)            5  cos(heading)     (heading is unbounded in env)
   6  fuel_remaining / reserve_capacity
   7  dist_to_goal / initial_dist
   8  wind_x / 30             9  wind_y / 30
  10  sin(bearing_error)     11  cos(bearing_error)   (goal bearing - heading)
  12-14 nearest no-fly zone : dx/30km, dy/30km, (dist - radius)/30km
  15-17 2nd-nearest zone    : same
  18  (altitude - cruise_alt) / cruise_alt

The no-fly-zone features give the agent explicit "sensor" awareness of
constraints, which is what lets the same policy generalise to new zone
layouts later (Weeks 9-10 robustness study).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

K_ZONES = 2          # number of nearest no-fly zones exposed to the agent
ZONE_SCALE = 30_000.0
FAR_ZONE = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # padding if < K zones


class UAVObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = 13 + 3 * K_ZONES
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        e = self.env.unwrapped
        x, y, alt, v, hdg, fuel, dist, wx, wy = obs
        goal_d0 = float(np.linalg.norm(e.goal_pos - e.start_pos)) or 1.0
        reserve = e.initial_mass * 0.5

        bearing = np.arctan2(e.goal_pos[1] - y, e.goal_pos[0] - x)
        b_err = bearing - hdg

        zones = []
        for (cx, cy, r) in e.no_fly_zones:
            dx, dy = cx - x, cy - y
            d = np.hypot(dx, dy)
            zones.append((d - r, dx, dy))
        zones.sort(key=lambda z: z[0])
        zone_feats = []
        for i in range(K_ZONES):
            if i < len(zones):
                clr, dx, dy = zones[i]
                zone_feats += [dx / ZONE_SCALE, dy / ZONE_SCALE, clr / ZONE_SCALE]
            else:
                zone_feats += list(FAR_ZONE)

        feats = np.array([
            x / goal_d0, y / ZONE_SCALE, alt / 15_000.0, (v - 100.0) / 150.0,
            np.sin(hdg), np.cos(hdg), fuel / reserve, dist / goal_d0,
            wx / 30.0, wy / 30.0, np.sin(b_err), np.cos(b_err),
            *zone_feats,
            (alt - e.cruise_alt) / e.cruise_alt,
        ], dtype=np.float32)
        return np.clip(feats, -5.0, 5.0)


class RewardScaleWrapper(gym.RewardWrapper):
    def __init__(self, env, scale: float = 0.1):
        super().__init__(env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale


class WindRandomizationWrapper(gym.Wrapper):
    """Samples wind speed in [0, max_speed] m/s and a random direction each episode."""

    def __init__(self, env, max_speed: float = 25.0):
        super().__init__(env)
        self.max_speed = max_speed

    def reset(self, **kwargs):
        rng = self.env.unwrapped.np_random
        s = rng.uniform(0.0, self.max_speed)
        th = rng.uniform(-np.pi, np.pi)
        self.env.unwrapped.wind_vector = np.array([s * np.cos(th), s * np.sin(th)], dtype=np.float32)
        return self.env.reset(**kwargs)


def make_env(seed: int = 0, reward_scale: float = 0.1, randomize_wind: bool = False,
             env_kwargs: dict | None = None, log_file: str | None = None):
    """Factory used by train.py / evaluate.py so training and evaluation see
    exactly the same wrapped environment."""
    from stable_baselines3.common.monitor import Monitor
    from uav_trajectory_env import UAVTrajectoryEnv

    def _init():
        env = UAVTrajectoryEnv(seed=seed, **(env_kwargs or {}))
        if randomize_wind:
            env = WindRandomizationWrapper(env)
        env = Monitor(env, filename=log_file, info_keywords=("fuel_used_total", "reached_goal"))
        env = UAVObsWrapper(env)
        env = RewardScaleWrapper(env, reward_scale)
        return env

    return _init

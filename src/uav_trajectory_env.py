"""
Step 1: Environment & Dynamics Setup
=====================================
Custom Gymnasium environment for UAV energy-optimal trajectory planning.

Implements:
  - Point-mass kinematics (position, altitude, velocity, heading)
  - Real fuel-burn model via OpenAP (Airbus A320 by default)
  - Static no-fly zones (circular)
  - Simple constant/gust wind field
  - MDP formulation: state, action, reward, termination

This is the Week 1-2 deliverable from the 12-week implementation plan.

v1.1 changes (made during Step 3, DRL training):
  - OpenAP `enroute()` expects true airspeed in KNOTS; v1.0 passed m/s. This
    units bug is what made high-altitude cruise look *more* expensive than
    low altitude (the "documented limitation" in v1.0). Fixed: tas * 1.94384.
  - Fuel model now uses the ACTUAL vertical speed after altitude limits, and
    includes acceleration (`acc`) so speeding up costs fuel.
  - Altitude floor is now a minimum safe altitude (default 500 m, matching the
    DP baseline's lowest level) instead of 0 m, and the ceiling is 12,000 m.
Run this file directly to sanity-check the environment and view a
random-policy trajectory plot.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import openap


class UAVTrajectoryEnv(gym.Env):
    """
    Point-mass UAV trajectory environment optimizing for fuel efficiency
    under static no-fly-zone constraints and a simple wind field.

    STATE  (9,): [x, y, altitude, velocity, heading,
                  fuel_remaining, dist_to_goal, wind_x, wind_y]
    ACTION (3,): [thrust_cmd, climb_rate_cmd, heading_rate_cmd]  in [-1, 1]
    REWARD:      -fuel_burn - time_penalty - constraint_violation
                 -altitude_deviation + progress_toward_goal
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        aircraft: str = "A320",
        dt: float = 5.0,               # seconds per simulation step
        max_steps: int = 400,
        start_pos: tuple = (0.0, 0.0),
        start_alt: float = 500.0,      # m
        goal_pos: tuple = (150_000.0, 0.0),   # 150 km away
        cruise_alt: float = 10_000.0,  # m, target cruise altitude
        min_alt: float = 500.0,        # m, minimum safe altitude (v1.1; was 0 = ground)
        max_alt: float = 12_000.0,     # m, service ceiling (A320 ~ 12,000 m)
        initial_mass: float = 60_000.0,  # kg
        no_fly_zones: list | None = None,   # list of (cx, cy, radius)
        wind_vector: tuple = (15.0, 0.0),   # m/s, constant wind (can randomize)
        reward_weights: dict | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        # ---- Aircraft performance model (OpenAP) ----
        self.fuelflow_model = openap.FuelFlow(ac=aircraft)
        self.thrust_model = openap.Thrust(ac=aircraft)
        self.aircraft_props = openap.prop.aircraft(aircraft)

        # ---- Simulation parameters ----
        self.dt = dt
        self.max_steps = max_steps
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.start_alt = start_alt
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.cruise_alt = cruise_alt
        self.min_alt = min_alt
        self.max_alt = max_alt
        self.initial_mass = initial_mass
        self.no_fly_zones = no_fly_zones or [
            (60_000.0, 8_000.0, 12_000.0),   # (center_x, center_y, radius) in meters
            (100_000.0, -10_000.0, 15_000.0),
        ]
        self.wind_vector = np.array(wind_vector, dtype=np.float32)

        # ---- Reward weights (tune these during training) ----
        default_weights = dict(
            w_fuel=1.0,
            w_time=0.05,
            w_violation=50.0,
            w_altitude=0.001,
            w_progress=0.01,
        )
        self.w = {**default_weights, **(reward_weights or {})}

        # ---- Gym spaces ----
        # obs: x, y, alt, v, heading, fuel_remaining, dist_to_goal, wind_x, wind_y
        obs_low = np.array([-1e6, -1e6, 0, 0, -np.pi, 0, 0, -100, -100], dtype=np.float32)
        obs_high = np.array([1e6, 1e6, 15000, 300, np.pi, initial_mass, 2e6, 100, 100], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # action: thrust_cmd, climb_rate_cmd, heading_rate_cmd, all normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.pos = self.start_pos.copy()
        self.alt = self.start_alt
        self.velocity = 120.0          # m/s, initial airspeed
        self.heading = 0.0             # radians, pointing toward +x
        self.mass = self.initial_mass
        self.fuel_used = 0.0
        self.t_step = 0
        self._prev_dist = float(np.linalg.norm(self.pos - self.goal_pos))

        obs = self._get_obs()
        info = {}
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        thrust_cmd, climb_rate_cmd, heading_rate_cmd = action

        # ---- Map normalized actions to physical ranges ----
        max_climb_rate = 15.0       # m/s
        max_heading_rate = 0.15     # rad/s
        climb_rate = climb_rate_cmd * max_climb_rate
        heading_rate = heading_rate_cmd * max_heading_rate

        # thrust_cmd in [-1,1] -> throttle fraction in [0.2, 1.0]
        throttle = 0.6 + 0.4 * thrust_cmd

        # ---- Point-mass kinematics update ----
        self.heading += heading_rate * self.dt
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi   # v1.1: keep in [-pi, pi]
        ground_vx = self.velocity * np.cos(self.heading) + self.wind_vector[0]
        ground_vy = self.velocity * np.sin(self.heading) + self.wind_vector[1]
        self.pos += np.array([ground_vx, ground_vy]) * self.dt
        prev_alt = self.alt
        self.alt = float(np.clip(self.alt + climb_rate * self.dt, self.min_alt, self.max_alt))
        # v1.1 fix: use the ACTUAL vertical speed after altitude limits. Previously the
        # commanded climb rate was passed to the fuel model even when the aircraft was
        # pinned at the altitude floor, so a DRL agent learned to "descend into the
        # ground" forever at near-idle fuel flow (reward hacking found in Step 3).
        actual_vs = (self.alt - prev_alt) / self.dt

        # simple speed response: throttle nudges airspeed toward a throttle-dependent target
        target_speed = 100.0 + throttle * 150.0
        prev_v = self.velocity
        self.velocity += (target_speed - self.velocity) * 0.1
        accel = (self.velocity - prev_v) / self.dt   # v1.1: accelerating now costs fuel

        # ---- Fuel burn via OpenAP ----
        fuel_flow = self.fuelflow_model.enroute(
            mass=self.mass,
            tas=self.velocity * 1.94384,   # OpenAP expects knots
            alt=self.alt * 3.28084,        # OpenAP expects feet
            vs=actual_vs * 196.85,         # m/s -> ft/min
            acc=accel,                     # m/s^2
        )
        fuel_burned = float(fuel_flow) * self.dt
        fuel_burned = max(fuel_burned, 0.0)
        self.fuel_used += fuel_burned
        self.mass = max(self.mass - fuel_burned, self.initial_mass * 0.5)

        # ---- Constraint violation check (no-fly zones) ----
        violation = 0.0
        for (cx, cy, r) in self.no_fly_zones:
            d = np.linalg.norm(self.pos - np.array([cx, cy]))
            if d < r:
                violation += (r - d) / r   # normalized penetration depth

        # ---- Reward computation ----
        dist_to_goal = np.linalg.norm(self.pos - self.goal_pos)
        prev_dist = getattr(self, "_prev_dist", dist_to_goal)
        progress = prev_dist - dist_to_goal
        self._prev_dist = dist_to_goal

        altitude_dev = abs(self.alt - self.cruise_alt) / self.cruise_alt

        reward = (
            - self.w["w_fuel"] * fuel_burned
            - self.w["w_time"] * self.dt
            - self.w["w_violation"] * violation
            - self.w["w_altitude"] * altitude_dev
            + self.w["w_progress"] * progress
        )

        # ---- Termination / truncation ----
        self.t_step += 1
        reached_goal = dist_to_goal < 3000.0
        out_of_fuel = self.mass <= self.initial_mass * 0.5 + 1.0
        terminated = bool(reached_goal or out_of_fuel)
        truncated = bool(self.t_step >= self.max_steps)

        if reached_goal:
            reward += 200.0   # bonus for successfully completing the route

        obs = self._get_obs()
        info = {
            "fuel_used_total": self.fuel_used,
            "dist_to_goal": dist_to_goal,
            "violation": violation,
            "reached_goal": reached_goal,
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        dist_to_goal = np.linalg.norm(self.pos - self.goal_pos)
        return np.array([
            self.pos[0], self.pos[1], self.alt, self.velocity, self.heading,
            self.mass - (self.initial_mass * 0.5),  # fuel remaining above reserve
            dist_to_goal, self.wind_vector[0], self.wind_vector[1],
        ], dtype=np.float32)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    import matplotlib.pyplot as plt

    env = UAVTrajectoryEnv(seed=42)

    # 1) Structural sanity check (validates Gym API compliance)
    check_env(env.unwrapped, skip_render_check=True)
    print("check_env passed: environment is Gymnasium-API compliant.\n")

    # 2) Run one episode with a naive scripted policy (climb, then cruise straight)
    obs, info = env.reset(seed=0)
    trajectory = [obs[:3].copy()]
    total_reward = 0.0

    for step in range(env.max_steps):
        # naive policy: full throttle, climb until near cruise alt, then level off, fly straight
        climb_cmd = 1.0 if obs[2] < env.cruise_alt - 200 else -0.1
        action = np.array([0.3, climb_cmd, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        trajectory.append(obs[:3].copy())
        if terminated or truncated:
            break

    trajectory = np.array(trajectory)
    print(f"Episode finished after {step + 1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total fuel used: {info['fuel_used_total']:.1f} kg")
    print(f"Reached goal: {info['reached_goal']}")
    print(f"Final distance to goal: {info['dist_to_goal']:.0f} m")

    # 3) Plot the resulting trajectory (top-down + altitude profile)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(trajectory[:, 0] / 1000, trajectory[:, 1] / 1000, color="#F4A825", lw=2)
    for (cx, cy, r) in env.no_fly_zones:
        circle = plt.Circle((cx / 1000, cy / 1000), r / 1000, color="red", alpha=0.3)
        axes[0].add_patch(circle)
    axes[0].scatter(*(env.goal_pos / 1000), color="green", marker="*", s=200, label="Goal")
    axes[0].scatter(*(env.start_pos / 1000), color="blue", marker="o", s=80, label="Start")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")
    axes[0].set_title("Top-down trajectory (red = no-fly zones)")
    axes[0].legend()
    axes[0].axis("equal")

    axes[1].plot(trajectory[:, 0] / 1000, trajectory[:, 2], color="#16213E", lw=2)
    axes[1].axhline(env.cruise_alt, color="gray", linestyle="--", label="Target cruise altitude")
    axes[1].set_xlabel("x (km)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].set_title("Altitude profile")
    axes[1].legend()

    plt.tight_layout()
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "sample_trajectory.png"), dpi=150)
    print("\nSaved plot to sample_trajectory.png")

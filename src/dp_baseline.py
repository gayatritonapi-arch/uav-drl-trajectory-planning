"""
Step 2: Classical Optimal-Control Baseline (Weeks 3-4)
=========================================================
Dynamic-programming (DP) solver for the fuel-optimal UAV trajectory
problem, built on a discretized layered graph over (downrange distance,
lateral offset, altitude). This mirrors the same aircraft, fuel model,
goal, and no-fly zones as `uav_trajectory_env.py`, so its fuel result is
directly comparable to the DRL agent trained later.

Method
------
This is a classic "layered-graph shortest path" DP, the same principle
used in real flight-profile optimizers (climb/cruise/descent profile
optimization via BADA-based DP). Since downrange distance only ever
increases, the graph is a Directed Acyclic Graph (DAG): we discretize
distance into x-layers, and at each layer allow a small set of
(lateral offset, altitude) states. We compute the minimum-fuel path
from goal back to start via backward value iteration (Bellman
recursion), then read off the optimal policy forward from the start.

Key difference vs. the DRL formulation
---------------------------------------
- No-fly-zone violations are treated as a HARD constraint (transition is
  simply disallowed) rather than a soft reward penalty. This guarantees
  zero violations by construction -- a natural advantage of classical DP
  when the environment is known and static ahead of time.
- Requires the environment model to be known in advance (zone locations,
  fuel model) and re-solved from scratch if anything changes -- this is
  exactly the limitation DRL is meant to address for the "adaptive,
  real-time" claim in the project.

Simplifying assumptions (documented, not hidden)
--------------------------------------------------
- Constant cruise speed and constant mass assumed while evaluating each
  segment's fuel flow (the DRL point-mass environment models these more
  dynamically). This keeps the DP tractable on a coarse grid.
- The lateral/altitude grid is coarse for runtime tractability; climb
  rates implied by adjacent-layer altitude jumps are idealized, not
  hard-constrained to the same max climb rate used in the RL environment.
  Refine the grid resolution if you need tighter physical realism.
"""

import numpy as np
import matplotlib.pyplot as plt
import openap

from uav_trajectory_env import UAVTrajectoryEnv


class DPTrajectoryBaseline:
    def __init__(
        self,
        env: UAVTrajectoryEnv,
        n_x_layers: int = 50,
        y_range: tuple = (-30_000.0, 30_000.0),
        n_y_bins: int = 13,
        alt_levels: tuple = (500, 2000, 3500, 5000, 6500, 8000, 10000),
        cruise_speed: float = 220.0,   # m/s, constant-speed assumption for DP
        max_y_jump_bins: int = 1,      # lateral maneuverability per x-step
        max_alt_jump_levels: int = 1,  # climb/descent maneuverability per x-step
    ):
        self.env = env
        self.aircraft = env.aircraft_props
        self.fuelflow_model = env.fuelflow_model
        self.no_fly_zones = env.no_fly_zones
        self.mass = env.initial_mass
        self.cruise_speed = cruise_speed

        self.goal_x = float(env.goal_pos[0])
        self.goal_y = float(env.goal_pos[1])
        self.start_alt = env.start_alt

        self.x_grid = np.linspace(0.0, self.goal_x, n_x_layers)
        self.y_grid = np.linspace(y_range[0], y_range[1], n_y_bins)
        self.alt_levels = np.array(alt_levels, dtype=np.float32)

        self.max_y_jump = max_y_jump_bins
        self.max_alt_jump = max_alt_jump_levels

        self.n_x = len(self.x_grid)
        self.n_y = len(self.y_grid)
        self.n_alt = len(self.alt_levels)

    # ------------------------------------------------------------------
    def _segment_intersects_no_fly_zone(self, p1, p2):
        """Check if line segment p1->p2 passes within radius of any no-fly zone."""
        p1, p2 = np.array(p1), np.array(p2)
        seg = p2 - p1
        seg_len_sq = np.dot(seg, seg)
        for (cx, cy, r) in self.no_fly_zones:
            c = np.array([cx, cy])
            if seg_len_sq < 1e-9:
                d = np.linalg.norm(p1 - c)
            else:
                t = np.clip(np.dot(c - p1, seg) / seg_len_sq, 0.0, 1.0)
                closest = p1 + t * seg
                d = np.linalg.norm(closest - c)
            if d < r:
                return True
        return False

    # ------------------------------------------------------------------
    def _segment_fuel_cost(self, x1, y1, alt1, x2, y2, alt2):
        """Fuel burned (kg) traversing one segment, using the OpenAP model."""
        horiz_dist = np.hypot(x2 - x1, y2 - y1)
        dt = horiz_dist / self.cruise_speed
        if dt < 1e-6:
            return 0.0
        vs_m_s = (alt2 - alt1) / dt
        vs_ft_min = vs_m_s * 196.85
        alt_ft = ((alt1 + alt2) / 2.0) * 3.28084
        fuel_flow = self.fuelflow_model.enroute(
            mass=self.mass, tas=self.cruise_speed, alt=alt_ft, vs=vs_ft_min
        )
        return max(float(fuel_flow) * dt, 0.0)

    # ------------------------------------------------------------------
    def solve(self):
        """Backward value iteration over the layered graph. Returns the
        optimal path and total fuel cost."""
        INF = float("inf")
        n_x, n_y, n_alt = self.n_x, self.n_y, self.n_alt

        # V[i, j, k] = min fuel-to-go from layer i, y-bin j, alt-level k
        V = np.full((n_x, n_y, n_alt), INF, dtype=np.float64)
        # next_ptr[i, j, k] = (j_next, k_next) chosen at optimum
        next_ptr = np.full((n_x, n_y, n_alt, 2), -1, dtype=np.int32)

        # ---- Terminal cost at the last layer: penalize distance from goal ----
        last = n_x - 1
        for j in range(n_y):
            for k in range(n_alt):
                y_val = self.y_grid[j]
                alt_val = self.alt_levels[k]
                lateral_err = abs(y_val - self.goal_y)
                alt_err = abs(alt_val - self.start_alt)  # descend back near start alt
                V[last, j, k] = 0.02 * lateral_err + 0.05 * alt_err

        # ---- Backward recursion ----
        for i in range(n_x - 2, -1, -1):
            x1 = self.x_grid[i]
            x2 = self.x_grid[i + 1]
            for j in range(n_y):
                y1 = self.y_grid[j]
                for k in range(n_alt):
                    alt1 = self.alt_levels[k]
                    best_cost = INF
                    best_next = (-1, -1)

                    j_lo, j_hi = max(0, j - self.max_y_jump), min(n_y - 1, j + self.max_y_jump)
                    k_lo, k_hi = max(0, k - self.max_alt_jump), min(n_alt - 1, k + self.max_alt_jump)

                    for jn in range(j_lo, j_hi + 1):
                        y2 = self.y_grid[jn]
                        for kn in range(k_lo, k_hi + 1):
                            alt2 = self.alt_levels[kn]

                            if self._segment_intersects_no_fly_zone((x1, y1), (x2, y2)):
                                continue  # hard constraint: forbidden transition

                            seg_cost = self._segment_fuel_cost(x1, y1, alt1, x2, y2, alt2)
                            total_cost = seg_cost + V[i + 1, jn, kn]

                            if total_cost < best_cost:
                                best_cost = total_cost
                                best_next = (jn, kn)

                    V[i, j, k] = best_cost
                    next_ptr[i, j, k] = best_next

        # ---- Forward reconstruction from the start state ----
        j0 = int(np.argmin(np.abs(self.y_grid - 0.0)))
        k0 = int(np.argmin(np.abs(self.alt_levels - self.start_alt)))

        if V[0, j0, k0] == INF:
            raise RuntimeError(
                "No feasible fuel-optimal path found from the start state "
                "under the given grid resolution and no-fly-zone constraints. "
                "Try increasing max_y_jump_bins / n_y_bins."
            )

        path = [(self.x_grid[0], self.y_grid[j0], self.alt_levels[k0])]
        j, k = j0, k0
        total_fuel = 0.0
        for i in range(n_x - 1):
            jn, kn = next_ptr[i, j, k]
            x1, y1, alt1 = self.x_grid[i], self.y_grid[j], self.alt_levels[k]
            x2, y2, alt2 = self.x_grid[i + 1], self.y_grid[jn], self.alt_levels[kn]
            total_fuel += self._segment_fuel_cost(x1, y1, alt1, x2, y2, alt2)
            path.append((x2, y2, alt2))
            j, k = jn, kn

        return np.array(path), total_fuel


# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = UAVTrajectoryEnv(seed=42)
    dp = DPTrajectoryBaseline(env)

    print("Solving fuel-optimal trajectory via dynamic programming...")
    path, total_fuel = dp.solve()
    print(f"\nDP baseline complete.")
    print(f"Total fuel used: {total_fuel:.1f} kg")
    print(f"Path waypoints: {len(path)}")

    # Verify zero no-fly-zone violations (hard constraint should guarantee this)
    violations = 0
    for i in range(len(path) - 1):
        if dp._segment_intersects_no_fly_zone(path[i][:2], path[i + 1][:2]):
            violations += 1
    print(f"No-fly-zone violations: {violations} (should be 0 by construction)")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(path[:, 0] / 1000, path[:, 1] / 1000, color="#16213E", lw=2, marker="o", markersize=3)
    for (cx, cy, r) in env.no_fly_zones:
        circle = plt.Circle((cx / 1000, cy / 1000), r / 1000, color="red", alpha=0.3)
        axes[0].add_patch(circle)
    axes[0].scatter(*(env.goal_pos / 1000), color="green", marker="*", s=200, label="Goal")
    axes[0].scatter(0, 0, color="blue", marker="o", s=80, label="Start")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")
    axes[0].set_title("DP baseline: fuel-optimal path (avoids no-fly zones)")
    axes[0].legend()
    axes[0].axis("equal")

    axes[1].plot(path[:, 0] / 1000, path[:, 2], color="#F4A825", lw=2, marker="o", markersize=3)
    axes[1].set_xlabel("x (km)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].set_title("DP baseline: altitude profile")

    plt.tight_layout()
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "dp_baseline_trajectory.png"), dpi=150)
    print("\nSaved plot to dp_baseline_trajectory.png")

    print(f"\n--- Comparison so far ---")
    print(f"Naive scripted policy (Step 1, flies THROUGH no-fly zones): ~1252 kg fuel, 0 zones avoided")
    print(f"DP baseline (Step 2, hard-avoids no-fly zones):             {total_fuel:.0f} kg fuel, {violations} violations")
    print("Note: not a fully fair comparison yet -- the naive policy ignored constraints entirely.")
    print("The real comparison will be: trained DRL agent vs. this DP baseline (Weeks 9-10).")

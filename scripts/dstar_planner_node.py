#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dstar_planner_node.py
---------------------
D* Lite path planner for a Dubins car.

- Replans on every SDF update (with cooldown)
- Follows path with pure pursuit
- If no path found, robot crawls forward toward goal while waiting for next SDF update
- No fallback navigation, no bug algorithm, no frontier exploration
"""

import os
import math
import heapq
import time
import yaml
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from example_interfaces.msg import Bool as BoolMsg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from refinecbf_ros2.msg import Array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_env_yaml(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    dom = cfg["state_domain"]
    lo  = np.array(dom["lo"],         dtype=np.float64)
    hi  = np.array(dom["hi"],         dtype=np.float64)
    res = np.array(dom["resolution"], dtype=np.int32)
    axes = [np.linspace(float(l), float(h), int(n)) for l, h, n in zip(lo, hi, res)]
    return {"axes": axes}


def parse_control_yaml(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    goals = [(float(g[0]), float(g[1])) for g in cfg["nominal"]["goals"]]
    lim   = cfg["limits"]
    return {
        "goals":     goals,
        "max_acc":   float(lim["max_acc"]),
        "min_acc":   float(lim["min_acc"]),
        "max_vel":   float(lim["max_vel"]),
        "min_vel":   float(lim["min_vel"]),
        "max_omega": float(lim["max_omega"]),
    }


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))


# ---------------------------------------------------------------------------
# D* Lite
# ---------------------------------------------------------------------------

class DStarLite:
    INF = float("inf")

    def __init__(self, xs, ys, safety_margin):
        self.xs  = xs
        self.ys  = ys
        self.Nx  = len(xs)
        self.Ny  = len(ys)
        self.dx  = float(xs[1] - xs[0]) if self.Nx > 1 else 1.0
        self.dy  = float(ys[1] - ys[0]) if self.Ny > 1 else 1.0
        self.safety_margin = safety_margin
        self.sdf_grid = None  # (Ny, Nx) ndarray, set externally

        d = math.hypot(self.dx, self.dy)
        self._nbrs = [
            (-1,  0, self.dy), ( 1,  0, self.dy),
            ( 0, -1, self.dx), ( 0,  1, self.dx),
            (-1, -1, d),       (-1,  1, d),
            ( 1, -1, d),       ( 1,  1, d),
        ]

    def w2g(self, x, y):
        ix = int(round((x - self.xs[0]) / self.dx))
        iy = int(round((y - self.ys[0]) / self.dy))
        return max(0, min(self.Nx-1, ix)), max(0, min(self.Ny-1, iy))

    def g2w(self, ix, iy):
        return float(self.xs[ix]), float(self.ys[iy])

    def _inb(self, ix, iy):
        return 0 <= ix < self.Nx and 0 <= iy < self.Ny

    def _obs(self, ix, iy):
        if self.sdf_grid is None or not self._inb(ix, iy):
            return True
        return float(self.sdf_grid[iy, ix]) < self.safety_margin

    def _cost(self, ix, iy, step):
        """Traversal cost — penalise cells close to obstacles."""
        clearance = max(float(self.sdf_grid[iy, ix]) - self.safety_margin, 1e-3)
        return step * (1.0 + 0.3 / clearance)

    def plan(self, start_xy, goal_xy):
        """
        Returns list of (x,y) world coords from start to goal, or [] if no path.
        Start cell is always treated as free (robot is physically there).
        """
        if self.sdf_grid is None:
            return []

        sx, sy = self.w2g(*start_xy)
        gx, gy = self.w2g(*goal_xy)

        # Snap goal to nearest free cell if needed
        if self._obs(gx, gy):
            free = self._nearest_free(gx, gy)
            if free is None:
                return []
            gx, gy = free

        if (sx, sy) == (gx, gy):
            return [self.g2w(gx, gy)]

        # D* Lite: backward search from goal to start
        INF = self.INF
        g   = np.full((self.Ny, self.Nx), INF)
        rhs = np.full((self.Ny, self.Nx), INF)
        rhs[gy, gx] = 0.0

        def h(ix, iy):
            return math.hypot((ix-sx)*self.dx, (iy-sy)*self.dy)

        def key(ix, iy):
            m = min(g[iy,ix], rhs[iy,ix])
            return (m + h(ix,iy), m)

        heap = [(*key(gx,gy), gx, gy)]
        in_heap = {(gx,gy)}

        for _ in range(self.Nx * self.Ny * 2):
            if not heap:
                break
            k0, k1, cx, cy = heapq.heappop(heap)
            in_heap.discard((cx,cy))

            ck = key(cx,cy)
            if (k0,k1) > ck:
                heapq.heappush(heap, (*ck, cx, cy))
                in_heap.add((cx,cy))
                continue

            if g[cy,cx] > rhs[cy,cx]:
                g[cy,cx] = rhs[cy,cx]
            else:
                g[cy,cx] = INF
                if (cx,cy) not in in_heap:
                    heapq.heappush(heap, (*key(cx,cy), cx, cy))
                    in_heap.add((cx,cy))

            # Early exit once start is settled
            if g[sy,sx] == rhs[sy,sx] and heap:
                sk = key(sx,sy)
                if (k0,k1) >= sk:
                    break

            for diy, dix, step in self._nbrs:
                nx, ny = cx+dix, cy+diy
                if not self._inb(nx,ny):
                    continue
                # Start cell is always passable
                if (nx,ny) == (sx,sy):
                    c = step
                elif self._obs(nx,ny):
                    continue
                else:
                    c = self._cost(nx,ny,step)
                nr = g[cy,cx] + c
                if nr < rhs[ny,nx]:
                    rhs[ny,nx] = nr
                    if (nx,ny) not in in_heap:
                        heapq.heappush(heap, (*key(nx,ny), nx, ny))
                        in_heap.add((nx,ny))

        if g[sy,sx] == INF:
            return []

        # Greedy traceback from start → goal
        path = []
        cx, cy = sx, sy
        seen = {(cx,cy)}
        for _ in range(self.Nx * self.Ny):
            path.append(self.g2w(cx,cy))
            if (cx,cy) == (gx,gy):
                break
            best, best_c = None, INF
            for diy, dix, step in self._nbrs:
                nx, ny = cx+dix, cy+diy
                if not self._inb(nx,ny) or (nx,ny) in seen:
                    continue
                if self._obs(nx,ny) and (nx,ny) != (sx,sy):
                    continue
                c = g[ny,nx] + step
                if c < best_c:
                    best_c = c
                    best = (nx,ny)
            if best is None:
                break
            seen.add(best)
            cx, cy = best

        return path

    def _nearest_free(self, ix, iy, max_r=20):
        q = deque([(ix, iy, 0)])
        seen = {(ix,iy)}
        while q:
            cx, cy, r = q.popleft()
            if r > max_r:
                return None
            if not self._obs(cx,cy):
                return cx, cy
            for diy, dix, _ in self._nbrs:
                nx, ny = cx+dix, cy+diy
                if self._inb(nx,ny) and (nx,ny) not in seen:
                    seen.add((nx,ny))
                    q.append((nx,ny,r+1))
        return None

    def path_still_valid(self, path):
        """Check that no waypoint in the next 10 steps is now an obstacle."""
        for wx, wy in path[:10]:
            ix, iy = self.w2g(wx, wy)
            if self._obs(ix,iy):
                return False
        return True

    def closest_reachable_free_to_goal(self, start_xy, goal_xy):
        """
        BFS outward from the robot's position over free cells, collecting all
        reachable free cells, then returning the one closest to the goal in
        world-space. This guarantees D* Lite can plan a path to the returned cell.
        Returns world (x, y) or None.
        """
        if self.sdf_grid is None:
            return None

        sx, sy = self.w2g(*start_xy)
        gx, gy = self.w2g(*goal_xy)

        # BFS over free cells reachable from start
        visited = {(sx, sy)}
        q = deque([(sx, sy)])
        reachable = []

        while q:
            cx, cy = q.popleft()
            reachable.append((cx, cy))
            for diy, dix, _ in self._nbrs:
                nx, ny = cx+dix, cy+diy
                if self._inb(nx, ny) and (nx,ny) not in visited and not self._obs(nx,ny):
                    visited.add((nx,ny))
                    q.append((nx,ny))

        if not reachable:
            return None

        # Return reachable free cell closest to goal
        best = min(reachable, key=lambda c: math.hypot(
            (c[0]-gx)*self.dx, (c[1]-gy)*self.dy))
        return self.g2w(*best)


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class DStarPlannerNode(Node):

    def __init__(self):
        super().__init__("dstar_planner_node")

        # Parameters
        for name, default in [
            ("env_config_path",     ""),
            ("control_config_path", ""),
            ("sdf_file_path",       ""),
        ]:
            try:
                self.declare_parameter(name, default)
            except Exception:
                pass  # already declared by parent class
        self.declare_parameter("odom_topic",          "/odom")
        self.declare_parameter("cmd_vel_topic",       "/cmd_vel")
        self.declare_parameter("sdf_update_topic",    "/env/sdf_update")
        self.declare_parameter("safety_margin",       0.3)
        self.declare_parameter("goal_tolerance",      0.4)
        self.declare_parameter("lookahead_dist",      1.0)
        self.declare_parameter("control_rate_hz",     30.0)
        self.declare_parameter("replan_cooldown",     1.0)
        self.declare_parameter("crawl_vel",           0.15)  # m/s when no path

        env_path      = self.get_parameter("env_config_path").value
        ctrl_path     = self.get_parameter("control_config_path").value
        self.sdf_path = self.get_parameter("sdf_file_path").value
        odom_topic    = self.get_parameter("odom_topic").value
        cmd_topic     = self.get_parameter("cmd_vel_topic").value
        sdf_topic     = self.get_parameter("sdf_update_topic").value

        self.safety_margin   = float(self.get_parameter("safety_margin").value)
        self.goal_tol        = float(self.get_parameter("goal_tolerance").value)
        self.lookahead_dist  = float(self.get_parameter("lookahead_dist").value)
        hz                   = float(self.get_parameter("control_rate_hz").value)
        self.replan_cooldown = float(self.get_parameter("replan_cooldown").value)
        self.crawl_vel       = float(self.get_parameter("crawl_vel").value)
        self.dt              = 1.0 / hz

        for p, label in [(env_path, "env_config_path"), (ctrl_path, "control_config_path")]:
            if not p or not os.path.exists(p):
                self.get_logger().fatal(f"{label} not found: '{p}'")
                raise RuntimeError(f"{label} required")

        env  = parse_env_yaml(env_path)
        ctrl = parse_control_yaml(ctrl_path)

        self.xs        = env["axes"][0]
        self.ys        = env["axes"][1]
        self.goals     = ctrl["goals"]
        self.max_acc   = ctrl["max_acc"]
        self.min_acc   = ctrl["min_acc"]
        self.max_vel   = ctrl["max_vel"]
        self.min_vel   = ctrl["min_vel"]
        self.max_omega = ctrl["max_omega"]

        self.planner = DStarLite(self.xs, self.ys, self.safety_margin)

        # State
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0
        self.robot_v     = 0.0
        self.odom_ready  = False

        self.goal_idx = 0
        self.path     = []

        self._last_replan_t = 0.0

        qos_rel  = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=5)
        qos_best = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)

        self.cmd_pub        = self.create_publisher(Twist,   cmd_topic, qos_rel)
        self.vf_pub         = self.create_publisher(Float32, "/path_planner_value_function", qos_rel)
        self.solve_time_pub = self.create_publisher(Float32, "/pp_controller_solve_time",    qos_rel)
        self.goal_reached_pub = self.create_publisher(BoolMsg, "/pp_goal_reached",           qos_rel)
        self.control_pub    = self.create_publisher(Array,   "/pp_control_commands",         qos_rel)
        self.state_pub      = self.create_publisher(Array,   "/pp_state",                    qos_rel)
        self.create_subscription(Odometry, odom_topic, self._on_odom, qos_best)
        self.create_subscription(BoolMsg, sdf_topic, self._on_sdf_update, qos_rel)
        self.create_timer(self.dt, self._control_tick)

        self.get_logger().info(
            f"DStarPlannerNode ready | goals={len(self.goals)} "
            f"safety_margin={self.safety_margin}m "
            f"lookahead={self.lookahead_dist}m "
            f"crawl_vel={self.crawl_vel}m/s")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self.robot_x     = float(p.x)
        self.robot_y     = float(p.y)
        self.robot_theta = yaw_from_quat(msg.pose.pose.orientation)
        self.robot_v     = float(msg.twist.twist.linear.x)
        self.odom_ready  = True

    def _on_sdf_update(self, msg):
        if not msg.data:
            return
        if not self.sdf_path or not os.path.exists(self.sdf_path):
            return

        try:
            raw = np.load(self.sdf_path)
            # File is saved as (Nx, Ny) — transpose to (Ny, Nx)
            if raw.shape == (len(self.xs), len(self.ys)):
                self.planner.sdf_grid = raw.T
            elif raw.shape == (len(self.ys), len(self.xs)):
                self.planner.sdf_grid = raw
            else:
                self.get_logger().warn(f"Unexpected SDF shape {raw.shape}")
                return
        except Exception as e:
            self.get_logger().error(f"Failed to load SDF: {e}")
            return

        if not self.odom_ready:
            return

        now = time.perf_counter()
        if now - self._last_replan_t < self.replan_cooldown:
            return
        self._last_replan_t = now
        self.get_logger().info("SDF updated — replanning")
        self._replan()

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _goal(self):
        return self.goals[self.goal_idx] if self.goal_idx < len(self.goals) else None

    def _replan(self):
        goal = self._goal()
        if goal is None:
            return

        start = (self.robot_x, self.robot_y)
        path = self.planner.plan(start, goal)

        if path:
            self.path = path
            self.get_logger().info(
                f"Path found to goal {self.goal_idx+1} "
                f"({goal[0]:.1f},{goal[1]:.1f}): {len(path)} waypoints")
            return

        # No path to goal — find the free cell closest to the goal that is
        # also reachable from the robot, and navigate there.
        # On each SDF update this replans, so the target advances as new
        # free space is revealed by the LiDAR.
        target = self.planner.closest_reachable_free_to_goal(
            start, goal)
        if target is not None:
            path = self.planner.plan(start, target)
            if path:
                self.path = path
                self.get_logger().info(
                    f"No path to goal {self.goal_idx+1} — navigating to closest "
                    f"reachable free point ({target[0]:.1f},{target[1]:.1f}), "
                    f"{math.hypot(target[0]-goal[0], target[1]-goal[1]):.1f}m from goal")
                return
            self.get_logger().warn("Frontier target plan failed unexpectedly")

        self.path = []
        self.get_logger().info(
            f"No path to goal {self.goal_idx+1} "
            f"({goal[0]:.1f},{goal[1]:.1f}) — crawling toward goal")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _control_tick(self):
        if not self.odom_ready:
            return

        goal = self._goal()
        if goal is None:
            self._stop()
            return

        # Goal reached
        dist = math.hypot(self.robot_x - goal[0], self.robot_y - goal[1])
        if dist < self.goal_tol:
            self.get_logger().info(f"Goal {self.goal_idx+1}/{len(self.goals)} reached!")
            self.goal_reached_pub.publish(BoolMsg(data=True))
            self.goal_idx += 1
            self.path = []
            if self.goal_idx >= len(self.goals):
                self.get_logger().info("All goals completed!")
                self._stop()
                return
            self._last_replan_t = 0.0
            self._replan()
            return

        # Invalidate path if obstacle appeared on it
        if self.path and not self.planner.path_still_valid(self.path):
            self.get_logger().info("Path invalidated — crawling toward goal until replan")
            self.path = []

        # No path — crawl forward toward goal while waiting for replan
        if not self.path:
            self._crawl_toward_goal(goal)
            return

        # Trim waypoints the robot has passed
        while len(self.path) > 1:
            if math.hypot(self.path[0][0] - self.robot_x,
                          self.path[0][1] - self.robot_y) < self.lookahead_dist * 0.5:
                self.path.pop(0)
            else:
                break

        # Find lookahead point on path
        lx, ly = self.path[-1]
        for wx, wy in self.path:
            if math.hypot(wx - self.robot_x, wy - self.robot_y) >= self.lookahead_dist:
                lx, ly = wx, wy
                break

        # PD_acc control law — identical gains to NominalControlPDAcc
        # omega: proportional to heading error toward lookahead point
        # acc:   proportional to (desired_vel - current_vel)
        # desired_vel: proportional to distance to goal, clipped to [min_vel, max_vel]
        tick_start = time.perf_counter()

        dx = lx - self.robot_x
        dy = ly - self.robot_y
        angle_to_target = math.atan2(dy, dx)
        angle_err = wrap(angle_to_target - self.robot_theta)

        desired_vel = float(np.clip(0.3 * dist, self.min_vel, self.max_vel))
        acc   = float(np.clip(0.5 * (desired_vel - self.robot_v), self.min_acc, self.max_acc))
        omega = float(np.clip(2.0 * angle_err, -self.max_omega, self.max_omega))

        v_new = float(np.clip(self.robot_v + acc * self.dt, self.min_vel, self.max_vel))
        self.robot_v = v_new

        solve_time = time.perf_counter() - tick_start

        # Publish value function (SDF at robot position)
        if self.planner.sdf_grid is not None:
            ix, iy = self.planner.w2g(self.robot_x, self.robot_y)
            vf_val = float(self.planner.sdf_grid[iy, ix])
            self.vf_pub.publish(Float32(data=vf_val))

        # Publish state [x, y, theta, v]
        state_msg = Array()
        state_msg.value = [self.robot_x, self.robot_y, self.robot_theta, self.robot_v]
        self.state_pub.publish(state_msg)

        # Publish solve time
        self.solve_time_pub.publish(Float32(data=float(solve_time * 1000.0)))  # ms

        # Publish control commands [acc, omega]
        ctrl_msg = Array()
        ctrl_msg.value = [acc, omega]
        self.control_pub.publish(ctrl_msg)

        cmd = Twist()
        cmd.linear.x  = v_new
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

    def _crawl_toward_goal(self, goal):
        """
        Steer directly toward the goal at a slow crawl velocity.
        Called when no planned path is available. The CBF safety filter
        upstream is expected to prevent actual collisions.
        """
        tick_start = time.perf_counter()

        dx = goal[0] - self.robot_x
        dy = goal[1] - self.robot_y
        angle_to_goal = math.atan2(dy, dx)
        angle_err = wrap(angle_to_goal - self.robot_theta)

        omega = float(np.clip(2.0 * angle_err, -self.max_omega, self.max_omega))
        acc   = float(np.clip(0.5 * (self.crawl_vel - self.robot_v), self.min_acc, self.max_acc))
        v_new = float(np.clip(self.robot_v + acc * self.dt, self.min_vel, self.max_vel))
        self.robot_v = v_new

        solve_time = time.perf_counter() - tick_start

        # Publish value function (SDF at robot position)
        if self.planner.sdf_grid is not None:
            ix, iy = self.planner.w2g(self.robot_x, self.robot_y)
            vf_val = float(self.planner.sdf_grid[iy, ix])
            self.vf_pub.publish(Float32(data=vf_val))

        # Publish state [x, y, theta, v]
        state_msg = Array()
        state_msg.value = [self.robot_x, self.robot_y, self.robot_theta, self.robot_v]
        self.state_pub.publish(state_msg)

        # Publish solve time
        self.solve_time_pub.publish(Float32(data=float(solve_time * 1000.0)))  # ms

        # Publish control commands [acc, omega]
        ctrl_msg = Array()
        ctrl_msg.value = [acc, omega]
        self.control_pub.publish(ctrl_msg)

        cmd = Twist()
        cmd.linear.x  = v_new
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

    def _stop(self):
        self.robot_v = 0.0
        self.cmd_pub.publish(Twist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = DStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
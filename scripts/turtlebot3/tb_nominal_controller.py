#!/usr/bin/env python3

import rclpy
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from hjr_nominal_control import NominalControlHJ
from pd_nominal_control import NominalControlPD
from pd_acc_nominal_control import NominalControlPDAcc
from refinecbf_ros2.srv import HighLevelCommand
from nominal_controller import NominalController
from config import Config

from refinecbf_ros2.msg import Array
from std_msgs.msg import Int32
from example_interfaces.msg import Bool


class TurtlebotNominalControl(NominalController):
    def __init__(self):
        super().__init__("tb_nominal_control")

        # === Parameters ===
        self.declare_parameter("controller_type", self.control_config.get("controller_type", "PD"))
        self.controller_type = self.get_parameter("controller_type").value
        self.declare_parameter("loop_goals", False)
        self.loop_goals = bool(self.get_parameter("loop_goals").value)

        # Build goals list from YAML
        self.goals = self._load_goals_from_config()
        if len(self.goals) == 0:
            raise RuntimeError("No goals found in control config (expected `nominal.goals` or `nominal.goal.coordinates`).")

        # Limits
        if self.controller_type == "PD_acc":
            self.max_acc = self.control_config["limits"]["max_acc"]
            self.min_acc = self.control_config["limits"]["min_acc"]
            self.max_omega = self.control_config["limits"]["max_omega"]
            self.umin = np.array([self.min_acc, -self.max_omega])
            self.umax = np.array([self.max_acc, self.max_omega])
        else:
            self.max_vel = self.control_config["limits"]["max_vel"]
            self.min_vel = self.control_config["limits"]["min_vel"]
            self.max_omega = self.control_config["limits"]["max_omega"]
            self.umin = np.array([self.min_vel, -self.max_omega])
            self.umax = np.array([self.max_vel, self.max_omega])

        # HJR config (if used)
        self._hjr_cfg = None
        if self.controller_type == "HJR":
            self._hjr_cfg = {
                "umax_hjr": np.array(self.env_config["control_space"]["hi"]),
                "umin_hjr": np.array(self.env_config["control_space"]["lo"]),
                "padding": np.array(self.control_config["nominal"]["goal"].get("padding", [0.0, 0.0, 0.0])),
                "max_time": self.control_config["nominal"]["goal"].get("max_time", 3.0),
                "time_intervals": self.control_config["nominal"]["goal"].get("time_intervals", 50),
                "solver_accuracy": self.control_config["nominal"]["goal"].get("solver_accuracy", "medium"),
            }
            assert self._hjr_cfg["solver_accuracy"] in ["low", "medium", "high", "very_high"]
            self.hj_dynamics = self.config.hj_dynamics
            self.grid = self.config.grid

        # --- Pub/Sub for multi-goal orchestration ---
        self.current_goal_pub = self.create_publisher(Array, "current_goal", 10)
        self.goals_reached_pub = self.create_publisher(Int32, "goals_reached", 10)
        self.goal_reached_sub = self.create_subscription(Bool, "goal_reached", self._goal_reached_cb, 10)
        self.current_goal_info = self.create_publisher(Array, "current_goal_info", 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

        # State
        self.goal_idx = 0
        self.goals_reached = 0
        self.target = self.goals[self.goal_idx]

        # Build and start controller
        self._build_controller_for_target(self.target)
        self._publish_current_goal()
        self.get_logger().info(
            f"Nominal controller started. Target #{self.goal_idx+1}/{len(self.goals)}: "
            f"[{self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}]"
        )
        self.start_controller()

    # ---------- Helpers ----------
    def timer_callback(self):
        msg = Array()
        msg.value = [float(self.target[0]), float(self.target[1]), float(self.target[2]), float(self.target[3] if len(self.target) > 3 else 0.0)]
        self.current_goal_info.publish(msg)

    def _load_goals_from_config(self):
        """
        Accepts either:
          control_config['nominal']['goals'] = [[x,y,theta], [x,y,theta], ...]
        or:
          control_config['nominal']['goal']['coordinates'] = [x,y,theta]
        """
        goals = []
        nominal_cfg = self.control_config.get("nominal", {})
        if "goals" in nominal_cfg and isinstance(nominal_cfg["goals"], (list, tuple)):
            for g in nominal_cfg["goals"]:
                arr = np.array(g, dtype=float).reshape(-1)
                if arr.size == 2:
                    arr = np.array([arr[0], arr[1], 0.0, 0.0], dtype=float)
                goals.append(arr[:4])
        else:
            coords = nominal_cfg.get("goal", {}).get("coordinates", None)
            if coords is not None:
                arr = np.array(coords, dtype=float).reshape(-1)
                if arr.size == 2:
                    arr = np.array([arr[0], arr[1], 0.0], dtype=float)
                goals.append(arr[:3])
        return goals

    def _publish_current_goal(self):
        msg = Array()
        msg.value = [float(self.target[0]), float(self.target[1]), float(self.target[2]), float(self.target[3] if len(self.target) > 3 else 0.0)]
        self.current_goal_pub.publish(msg)

    def _publish_goals_reached(self):
        self.goals_reached_pub.publish(Int32(data=int(self.goals_reached)))

    def _build_controller_for_target(self, target_np):
        if self.controller_type == "HJR":
            cfg = self._hjr_cfg
            controller_prep = NominalControlHJ(
                self.hj_dynamics,
                self.grid,
                final_time=cfg["max_time"],
                time_intervals=cfg["time_intervals"],
                solver_accuracy=cfg["solver_accuracy"],
                target=target_np,
                padding=cfg["padding"],
                progress_bar=True,
            )
            self.get_logger().info("Solving for nominal control with HJR backend...")
            self.controller = controller_prep.get_nominal_control

        elif self.controller_type == "PD":
            self.controller = NominalControlPD(target=target_np, umin=self.umin, umax=self.umax).get_nominal_control

        elif self.controller_type == "PD_acc":
            self.controller = NominalControlPDAcc(target=target_np, umin=self.umin, umax=self.umax).get_nominal_control

        else:
            raise NotImplementedError(f"{self.controller_type} is not a valid controller type")

    # ---------- Callbacks ----------

    def _goal_reached_cb(self, msg: Bool):
        if not msg.data:
            return

        self.goals_reached += 1
        self._publish_goals_reached()

        # If last goal reached
        if self.goal_idx >= len(self.goals) - 1:
            if self.loop_goals and len(self.goals) > 0:
                self.goal_idx = 0
                self.get_logger().info("Looping goals: restarting from first goal.")
            else:
                self.get_logger().info("All goals reached. Holding position.")
                return
        else:
            self.goal_idx += 1

        # Update target and rebuild controller
        self.target = self.goals[self.goal_idx]
        self._build_controller_for_target(self.target)
        self._publish_current_goal()
        self.get_logger().info(
            f"Switched to next target #{self.goal_idx+1}/{len(self.goals)}: "
            f"[{self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}, {self.target[3]:.3f}]"
        )
        self.start_controller()



def main(args=None):
    rclpy.init(args=args)
    controller = TurtlebotNominalControl()
    try:
        rclpy.spin(controller)
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
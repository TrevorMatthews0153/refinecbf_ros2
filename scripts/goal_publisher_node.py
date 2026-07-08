#!/usr/bin/env python3

import rclpy
import numpy as np
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from refinecbf_ros2.msg import Array
from refinecbf_ros2.srv import HighLevelCommand
from std_msgs.msg import Float32
from example_interfaces.msg import Bool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from template.hw_interface import BaseInterface
from utils import load_parameters
import rowan


class TurtlebotInterface(BaseInterface):
    """
    Converts state & control for TurtleBot, computes distance-to-goal,
    and publishes goal-reached flag for multi-goal orchestration.
    """

    state_msg_type = Odometry
    control_out_msg_type = Twist
    external_control_msg_type = Twist

    def __init__(self):
        super().__init__("turtlebot_interface")

        control_config = load_parameters(
            self.get_parameter("robot").value,
            self.get_parameter("exp").value,
            "control"
        )

        self.declare_parameters(
            "",
            [
                ("buffer_time_external_control", control_config["external"]["buffer_time"]),
                ("buffer_time_mod_external_control", control_config["external"]["mod_buffer_time"]),
                ("limits.max_vel", control_config["limits"]["max_vel"]),
                ("limits.min_vel", control_config["limits"]["min_vel"]),
                ("limits.max_acc", control_config["limits"]["max_acc"]),
                ("limits.min_acc", control_config["limits"]["min_acc"]),
                ("limits.max_omega", control_config["limits"]["max_omega"]),
                ("controller_type", control_config["controller_type"]),
                ("target", control_config["nominal"]["goal"]["coordinates"]),
                ("goal_reached_threshold", float(control_config["nominal"]["goal"].get("threshold", 0.20))),
            ]
        )

        # External control tracking
        self.external_control = None
        self.buffer_time_external_control = float(self.get_parameter("buffer_time_external_control").value)
        self.buffer_time_mod_external_control = float(self.get_parameter("buffer_time_mod_external_control").value)
        self.external_control_ts = None
        self.external_control_mod_ts = None

        # Limits & control type
        self.max_vel = float(self.get_parameter("limits.max_vel").value)
        self.min_vel = float(self.get_parameter("limits.min_vel").value)
        self.max_acc = float(self.get_parameter("limits.max_acc").value)
        self.min_acc = float(self.get_parameter("limits.min_acc").value)
        self.max_omega = float(self.get_parameter("limits.max_omega").value)
        self.controller_type = str(self.get_parameter("controller_type").value)

        # Target / goal
        self.target = np.array(self.get_parameter("target").value, dtype=float).reshape(-1)
        if self.target.size == 2:
            self.target = np.array([self.target[0], self.target[1], 0.0], dtype=float)
        self.goal_thresh = float(self.get_parameter("goal_reached_threshold").value)

        # Publishers for distance & goal-reached
        self.goal_reached_pub = self.create_publisher(Bool, "goal_reached", 10)          # <-- example_interfaces/Bool
        self.goal_distance_pub = self.create_publisher(Float32, "distance_to_goal", 10)

        # Subscribe to current goal from nominal controller (Array)
        self.create_subscription(Array, "current_goal", self._cb_current_goal, 10)       # <-- Array

        # State for control
        self.is_running = False
        self.init_subscribers()

        self.last_t = time.time()
        self.current_v = 0.0
        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        # Per-goal "edge trigger"
        self._goal_ack_sent = False

    # ---------- Services ----------

    def handle_high_level_command(self, request, response):
        if request.command == "start":
            if self.is_running:
                response.response = "Already running (start command ignored)"
            else:
                self.is_running = True
                response.response = "Turtlebot can move now"
        elif request.command == "end":
            if self.is_running:
                self.is_running = False
                response.response = "Turtlebot stopping"
            else:
                response.response = "Already stopped (end command ignored)"
        elif request.command == "goto":
            response.response = "goto command not implemented"
        else:
            response.response = f"actions not implemented ({request.command} command ignored)"
        return response

    # ---------- Subscriptions ----------

    def _cb_current_goal(self, msg: Array):
        # Expect [x, y, theta] in Array.value
        vals = np.array(msg.value, dtype=float).reshape(-1)
        if vals.size == 2:
            vals = np.array([vals[0], vals[1], 0.0], dtype=float)
        self.target = vals[:3]
        self._goal_ack_sent = False
        self.get_logger().info(f"New current goal received: [{self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}]")

    def callback_state(self, state_in_msg: Odometry):
        # Pose & twist extraction
        w = state_in_msg.pose.pose.orientation.w
        xq = state_in_msg.pose.pose.orientation.x
        yq = state_in_msg.pose.pose.orientation.y
        zq = state_in_msg.pose.pose.orientation.z
        v = state_in_msg.twist.twist.linear.x

        # Yaw
        euler = rowan.to_euler([xq, yq, zq, w])
        yaw = float(euler[2])

        self.current_x = float(state_in_msg.pose.pose.position.x)
        self.current_y = float(state_in_msg.pose.pose.position.y)
        self.current_yaw = yaw
        self.current_v = float(v)

        # Publish simplified state for safety node(s)
        state_out_msg = Array()
        state_out_msg.value = [self.current_x, self.current_y, yaw, self.current_v]
        self.state_pub.publish(state_out_msg)

        # --- Distance/threshold check ---
        if self.target is not None and np.isfinite(self.current_x) and np.isfinite(self.current_y):
            dist = float(np.linalg.norm(np.array([self.current_x, self.current_y]) - self.target[:2]))
            self.goal_distance_pub.publish(Float32(data=dist))

            if (dist <= self.goal_thresh) and not self._goal_ack_sent:
                self.goal_reached_pub.publish(Bool(data=True))   # <-- example_interfaces/Bool
                self._goal_ack_sent = True
                self.get_logger().info(f"Goal reached within {self.goal_thresh:.2f} m (dist={dist:.3f}).")

    # ---------- Controls ----------

    def process_safe_control(self, control_in_msg):
        """
        Convert safety-filtered control (Array) into Twist for the robot.
        Also enforces crisp stop when goal is reached.
        """
        if self.controller_type == "PD_acc":
            t = time.time()
            dt = max(1e-3, t - self.last_t)
            self.last_t = t

            acc = float(control_in_msg.value[0])
            v_next = self.current_v + acc * dt

            cmd = self.control_out_msg_type()
            cmd.linear.x = float(np.clip(v_next, 0.0, self.max_vel))
            cmd.linear.y = 0.0
            cmd.linear.z = 0.0
        else:
            ulin = float(control_in_msg.value[0])
            cmd = self.control_out_msg_type()
            cmd.linear.x = float(np.clip(ulin, self.min_vel, self.max_vel))
            cmd.linear.y = 0.0
            cmd.linear.z = 0.0

        uomega = float(control_in_msg.value[1])
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = float(np.clip(uomega, -self.max_omega, self.max_omega))

        # If goal reached, hold still
        if self._goal_ack_sent:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def process_external_control(self, control_in_msg: Twist):
        """
        Nominal control arrives as Twist; convert to Array for safety node.
        Track timing for override logic.
        """
        arr = Array()
        arr.value = [float(control_in_msg.linear.x), float(control_in_msg.angular.z)]
        new_val = np.array(arr.value, dtype=float)

        now = time.time()
        if (self.external_control is None) or (not np.allclose(self.external_control, new_val, atol=1e-1, rtol=1e-1)):
            self.external_control_mod_ts = now
            self.external_control = new_val
        self.external_control_ts = now
        return arr

    def process_disturbance(self, disturbance_msg):
        raise NotImplementedError("Override to process the disturbance message")

    def override_nominal_control(self):
        """
        Decide whether external (teleop) should override the nominal control.
        """
        if self.external_control_ts is None or self.external_control_mod_ts is None:
            return False
        now = time.time()
        return (
            (now - self.external_control_ts) <= self.buffer_time_external_control
            and (now - self.external_control_mod_ts) <= self.buffer_time_mod_external_control
        )


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

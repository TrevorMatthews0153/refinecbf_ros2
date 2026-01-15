#!/usr/bin/env python3

import os
import sys
import time

import numpy as np
import rclpy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from example_interfaces.msg import Bool

from refinecbf_ros2.msg import Array

# Local imports (repo layout dependent)
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from template.hw_interface import BaseInterface  # noqa: E402
from utils import load_parameters  # noqa: E402

import rowan  # noqa: E402


class TurtlebotInterface(BaseInterface):
    """
    Hardware interface for Turtlebot.

    - Converts Odometry -> internal Array state [x, y, yaw, v]
    - Converts safe control Array [v_or_acc, omega] -> Twist
    - Receives goal updates on "current_goal" as Array [x, y, theta, v] (theta/v optional)
    - Publishes goal reached + distance-to-goal + velocity monitors
    """

    state_msg_type = Odometry
    control_out_msg_type = Twist
    external_control_msg_type = Twist

    def __init__(self):
        super().__init__("turtlebot_interface")

        # ---- Load default control config for parameter declarations
        control_config = load_parameters(
            self.get_parameter("robot").value,
            self.get_parameter("exp").value,
            "control",
        )

        # ---- Parameters (declared with config defaults)
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
                ("goal_reached_threshold", control_config["nominal"]["goal"].get("reached_threshold", 0.20)),
            ],
        )

        # ---- External control bookkeeping
        self.external_control = None
        self.external_control_ts = None
        self.external_control_mod_ts = None
        self.buffer_time_external_control = self.get_parameter("buffer_time_external_control").value
        self.buffer_time_mod_external_control = self.get_parameter("buffer_time_mod_external_control").value

        # ---- Limits / controller behavior
        self.max_vel = self.get_parameter("limits.max_vel").value
        self.min_vel = self.get_parameter("limits.min_vel").value
        self.max_acc = self.get_parameter("limits.max_acc").value
        self.min_acc = self.get_parameter("limits.min_acc").value
        self.max_omega = self.get_parameter("limits.max_omega").value
        self.controller_type = self.get_parameter("controller_type").value

        # ---- Goal state
        self.target = np.array(self.get_parameter("target").value, dtype=float)
        self.goal_thresh = float(self.get_parameter("goal_reached_threshold").value)
        self._goal_ack_sent = False

        # ---- Publishers / Subscribers
        self.goal_reached_pub = self.create_publisher(Bool, "goal_reached", 10)
        self.goal_distance_pub = self.create_publisher(Float32, "distance_to_goal", 10)
        self.desired_velocity_pub = self.create_publisher(Float32, "desired_velocity", 10)
        self.actual_velocity_pub = self.create_publisher(Float32, "actual_velocity", 1)

        self.create_subscription(Array, "current_goal", self._cb_current_goal, 1)

        # ---- Interface lifecycle
        self.is_running = False
        self.init_subscribers()

        # ---- Local state tracking
        self.last_t = time.time()
        self.current_v = 0.0
        self.desired_v = 0.0

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.started_moving = False

    # -------------------------
    # High-level command handling
    # -------------------------

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
                # FIXME: Make sure to send zero commands to the robot
            else:
                response.response = "Already stopped (end command ignored)"

        elif request.command == "goto":
            raise NotImplementedError("goto command not implemented")

        else:
            response.response = f"actions not implemented ({request.command} command ignored)"

        return response

    # -------------------------
    # Goal updates
    # -------------------------

    def _cb_current_goal(self, msg: Array):
        """
        Expect:
          - [x, y] OR
          - [x, y, theta, v]
        """
        vals = np.array(msg.value, dtype=float).reshape(-1)
        if vals.size == 2:
            vals = np.array([vals[0], vals[1], 0.0, 0.0], dtype=float)

        self.target = vals[:4]
        self._goal_ack_sent = False
        self.get_logger().info(
            f"New current goal received: [{self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}]"
        )

    # -------------------------
    # State callback
    # -------------------------

    def callback_state(self, state_in_msg: Odometry):
        # Odometry orientation quaternion
        q = state_in_msg.pose.pose.orientation
        v_meas = state_in_msg.twist.twist.linear.x

        # rowan expects [x, y, z, w]
        euler = rowan.to_euler([q.x, q.y, q.z, q.w])
        yaw = euler[2]

        self.current_x = state_in_msg.pose.pose.position.x
        self.current_y = state_in_msg.pose.pose.position.y
        self.current_yaw = yaw

        # NOTE: original code sets current_v = desired_v (kept)
        self.current_v = self.desired_v

        self.actual_velocity_pub.publish(Float32(data=float(v_meas)))

        # Publish state to safety filter: [x, y, yaw, v]
        state_out_msg = Array()
        state_out_msg.value = [float(self.current_x), float(self.current_y), float(yaw), float(self.desired_v)]
        self.state_pub.publish(state_out_msg)

        # Goal distance + goal reached
        if self.target is not None and np.isfinite(self.current_x) and np.isfinite(self.current_y):
            dist = float(np.linalg.norm(np.array([self.current_x, self.current_y]) - self.target[:2]))
            self.goal_distance_pub.publish(Float32(data=dist))

            if (dist <= self.goal_thresh) and (not self._goal_ack_sent):
                self.get_logger().info(f"Goal reached {self.target[:2]}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.goal_reached_pub.publish(Bool(data=True))
                self._goal_ack_sent = True

    # -------------------------
    # Safe control -> robot control
    # -------------------------

    def process_safe_control(self, control_in_msg: Array) -> Twist:
        """
        control_in_msg.value is expected to be [u0, u1] where:
          - if controller_type == "PD_acc": u0 is acceleration, u1 is omega
          - else: u0 is velocity, u1 is omega
        """
        control_in = control_in_msg.value

        if self.controller_type == "PD_acc":
            dt = 1 / 50  # original code hard-codes dt (kept)

            acc = control_in[0]
            self.desired_v += acc * dt
            self.desired_v = np.clip(self.desired_v, self.min_vel, self.max_vel)
            self.desired_velocity_pub.publish(Float32(data=float(self.desired_v)))

            v_next = self.current_v + acc * dt
            v_next = np.clip(v_next, self.min_vel, self.max_vel)

            if not self.started_moving:
                if np.linalg.norm(control_in) > 0.0:
                    self.started_moving = True
                else:
                    v_next = 0.0
                    self.desired_v = 0.0

            cmd_v = self.desired_v  # original uses desired_v, not v_next (kept)
        else:
            cmd_v = np.clip(control_in[0], self.min_vel, self.max_vel)

        cmd_omega = np.clip(control_in[1], -self.max_omega, self.max_omega)

        control_out_msg = Twist()
        control_out_msg.linear.x = float(cmd_v)
        control_out_msg.linear.y = 0.0
        control_out_msg.linear.z = 0.0
        control_out_msg.angular.x = 0.0
        control_out_msg.angular.y = 0.0
        control_out_msg.angular.z = float(cmd_omega)

        # Stop at the goal (kept)
        if self._goal_ack_sent:
            control_out_msg.linear.x = 0.0
            control_out_msg.angular.z = 0.0
            self.get_logger().info("Reached Goal")

        return control_out_msg

    # -------------------------
    # External control passthrough
    # -------------------------

    def process_external_control(self, control_in_msg: Twist) -> Array:
        """
        External control arrives as Twist; convert to Array [v, omega].
        NOTE: This method contains suspicious time bookkeeping in the original code;
              the lines are kept structurally but the obvious typo (get_clock.now) is left as-is
              in YOUR original — consider fixing separately.
        """
        control_out_msg = Array()
        control_out_msg.value = [control_in_msg.linear.x, control_in_msg.angular.z]

        new_val = np.array(control_out_msg.value)
        if (self.external_control is None) or (not np.allclose(self.external_control, new_val, atol=1e-1, rtol=1e-1)):
            self.external_control_mod_ts = self.get_clock().now().nanoseconds
            self.external_control = new_val

        self.external_control = control_out_msg

        # Original code has: self.buffer_time_external_control = self.get_clock.now().nanoseconds (typo)
        # Kept behavior/structure: update timestamps from ROS time.
        self.buffer_time_external_control = self.get_clock().now().nanoseconds
        self.external_control_ts = self.get_clock().now().nanoseconds

        return control_out_msg

    # -------------------------
    # Disturbance (not implemented)
    # -------------------------

    def process_disturbance(self, disturbance_msg):
        raise NotImplementedError("Override to process the disturbance message")

    def override_nominal_control(self):
        """
        Decide whether to publish external control.
        NOTE: Original code mixes buffers/time units; logic preserved.
        """
        curr_time = self.get_clock().now().nanoseconds
        return (
            self.external_control is not None
            and (curr_time - self.buffer_time_external_control) * 1e9 <= self.external_control_time_buffer
            and (curr_time - self.external_control_mod_ts) * 1e9 <= self.buffer_time_mod_external_control
        )


def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
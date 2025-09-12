#!/usr/bin/env python3

import rclpy
import numpy as np

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
from ament_index_python.packages import get_package_share_directory
import yaml
import time
from utils import load_parameters
import rowan


class TurtlebotInterface(BaseInterface):
    """
    This class converts the state and control messages from the SafetyFilterNode to the correct type
    for the Turtlebots.
    Each HW platform should have its own Interface node
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
                ("goal_reached_threshold", control_config["nominal"]["goal"].get("reached_threshold", 0.20)),  # meters
            ]
        )

        # Initialize external control parameters
        self.external_control = None
        self.buffer_time_external_control = self.get_parameter("buffer_time_external_control").value  # seconds
        self.buffer_time_mod_external_control = self.get_parameter("buffer_time_mod_external_control").value  # seconds
        self.external_control_ts = None
        self.external_control_mod_ts = None

        self.max_vel = self.get_parameter("limits.max_vel").value
        self.min_vel = self.get_parameter("limits.min_vel").value
        self.max_acc = self.get_parameter("limits.max_acc").value
        self.min_acc = self.get_parameter("limits.min_acc").value
        self.max_omega = self.get_parameter("limits.max_omega").value
        self.controller_type = self.get_parameter("controller_type").value


        self.target = np.array(self.get_parameter("target").value)
        self.goal_thresh = self.get_parameter("goal_reached_threshold").value

        self.goal_reached_pub = self.create_publisher(Bool, "goal_reached", 10)
        self.goal_distance_pub = self.create_publisher(Float32, "distance_to_goal", 10)
        self.create_subscription(Array, "current_goal", self._cb_current_goal, 10)

        self.is_running = False
        self.init_subscribers()

        self.last_t = self.get_clock().now() # keep track of previous timestamp for acceleration control
        self.current_v = 0.0 # keep track of current velocity for acceleration control
        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self._goal_ack_sent = False

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
            response.response = "actions not implemented ({} command ignored)".format(request.command)
        return response

    def _cb_current_goal(self, msg: Array):
        # Expect [x, y, theta, v] in Array.value
        vals = np.array(msg.value, dtype=float).reshape(-1)
        if vals.size == 2:
            vals = np.array([vals[0], vals[1], 0.0, 0.0], dtype=float)
        self.target = vals[:4]
        self._goal_ack_sent = False
        self.get_logger().info(f"New current goal received: [{self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}]")

    def callback_state(self, state_in_msg):
        w = state_in_msg.pose.pose.orientation.w
        x = state_in_msg.pose.pose.orientation.x
        y = state_in_msg.pose.pose.orientation.y
        z = state_in_msg.pose.pose.orientation.z
        v = state_in_msg.twist.twist.linear.x

        # Convert Quaternion to Yaw
        # yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (np.power(y, 2) + np.power(z, 2))) + np.pi / 2 # FIXME: why is this necessary? I think it has something to do with the odom and rviz coordinate frames
        # yaw = np.arctan2(np.sin(yaw),np.cos(yaw)) # Remap yaw to -pi to pi range
        euler = rowan.to_euler([x, y, z, w])
        # self.get_logger().info("Yaw: {:.2f}".format(euler[2]))
        # self.get_logger().info(f"Position: x={state_in_msg.pose.pose.position.x:.2f}, y={state_in_msg.pose.pose.position.y:.2f}, Yaw: {euler[2]:.2f}, v: {v:.2f}")
        yaw = euler[2]
        self.current_x = state_in_msg.pose.pose.position.x
        self.current_y = state_in_msg.pose.pose.position.y
        self.current_yaw = yaw
        self.current_v = v
        state_out_msg = Array()
        state_out_msg.value = [state_in_msg.pose.pose.position.x, state_in_msg.pose.pose.position.y, yaw, v]
        self.state_pub.publish(state_out_msg)
        if self.target is not None and np.isfinite(self.current_x) and np.isfinite(self.current_y):
            dist = float(np.linalg.norm(np.array([self.current_x, self.current_y]) - self.target[:2]))
            self.goal_distance_pub.publish(Float32(data=dist))

            if (dist <= self.goal_thresh) and (not self._goal_ack_sent):
                self.get_logger().info(f"Goal reached {self.target[:2]}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.goal_reached_pub.publish(Bool(data=True))
                self._goal_ack_sent = True

    def process_safe_control(self, control_in_msg):
        if self.controller_type =="PD_acc":
            #compute velocity control from acceleration control
            # t = time.time()
            # dt = max(t - self.last_t, 1/20)
            # self.last_t = t
            # get time stamp from control_in_msg header when available

            # now = self.get_clock().now()
            # dt = max((now - self.last_t).nanoseconds * 1e-9, 0.0)
            # self.last_t = now
            self.target_rate_hz = 20.0  # Hz
            now = self.get_clock().now()  # ROS time
            dt = 1.0/self.target_rate_hz if self.last_t is None else (now - self.last_t).nanoseconds * 1e-9
            dt = max(min(dt, 0.2), 1.0/(self.target_rate_hz*2))  # clamp
            self.last_t = now

            control_in = control_in_msg.value
            acc = control_in[0]
            
            # dv = np.clip(acc * dt, self.min_acc * dt, self.max_acc * dt) #FIXME: Should we clip the acceleration?
            v_next = self.current_v + acc * dt
            # v_next = self.current_v + dv
            v_next = np.clip(v_next, self.min_vel, self.max_vel)
            
            if np.linalg.norm(control_in) == 0.0:
                v_next = 0.0

            control_out_msg = self.control_out_msg_type()
            control_out_msg.linear.x = v_next  # FIXME: Ideally self.min_vel
            control_out_msg.linear.y = 0.0
            control_out_msg.linear.z = 0.0
        else:
            control_in = control_in_msg.value
            control_out_msg = self.control_out_msg_type()
            control_out_msg.linear.x = np.clip(control_in[0], self.min_vel, self.max_vel)
            control_out_msg.linear.y = 0.0
            control_out_msg.linear.z = 0.0

        control_out_msg.angular.x = 0.0
        control_out_msg.angular.y = 0.0
        control_out_msg.angular.z = np.clip(control_in[1], -self.max_omega, self.max_omega)
        # self.get_logger().info(f"Control omega: {control_out_msg.angular.z}")

        # Stop at the goal
        if self._goal_ack_sent:
                control_out_msg.linear.x = 0.0
                control_out_msg.angular.z = 0.0
                self.get_logger().info("Reached Goal")

        # self.get_logger().info(f"Control linear: {control_out_msg.linear.x}, Control angular: {control_out_msg.angular.z}")
        return control_out_msg

    def process_external_control(self, control_in_msg):
        # When nominal control comes through the HW interface, it is a Twist message
        control_out_msg = Array()
        control_out_msg.value = [control_in_msg.linear.x, control_in_msg.angular.z]
        new_val = np.array(control_out_msg.value)
        if (self.external_control is None) or (not np.allclose(self.external_control, new_val, atol=1e-1, rtol=1e-1)):
            # If the external control has changed, then reset the external control mod timestamp
            self.external_control_mod_ts = self.get_clock().now().nanoseconds
            self.external_control = new_val
        
        self.external_control = control_out_msg
        self.buffer_time_external_control = self.get_clock.now().nanoseconds
        self.external_control_ts = self.get_clock().now().nanoseconds
        return control_out_msg
    
    def process_disturbance(self, disturbance_msg):
        disturbance_in = disturbance_msg.value
        disturbance_out_msg = self.disturbance_out_msg_type()
        raise NotImplementedError("Override to process the disturbance message")
        return disturbance_out_msg

    def override_nominal_control(self):
        curr_time = self.get_clock().now().nanoseconds

        # Determine if external control should be published
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
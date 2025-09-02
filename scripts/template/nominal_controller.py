#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from refinecbf_ros2.msg import Array
from example_interfaces.msg import Bool
from utils import load_parameters
from config import Config


class NominalController(Node):
    """
    This class determines the non-safe (aka nominal) control of a robot using hj_reachability in refineCBF.

    Subscribers:
    - state_sub (~topics/state): Subscribes to the robot's state.

    Publishers:
    - control_pub (~topics/nominal_control): Publishes the nominal control.
    """

    def __init__(self, node_name, hj_setup=False):
        super().__init__(node_name)
        print(hj_setup)
        self.config = Config(self, hj_setup=hj_setup)
        # Get topics from parameters
        self.declare_parameters(
            "",
            [
                ("topics.cbf_state", rclpy.Parameter.Type.STRING),
                ("topics.cbf_nominal_control", rclpy.Parameter.Type.STRING),
            ],
        )
        state_topic = self.get_parameter("topics.cbf_state").value
        nominal_control_topic = self.get_parameter("topics.cbf_nominal_control").value

        # Initialize subscribers and publishers
        self.state_sub = self.create_subscription(Array, state_topic, self.callback_state, 1)
        self.control_pub = self.create_publisher(Array, nominal_control_topic, 1)

        self.state = None
        self.controller = None

        # Initialize control variables
        self.control_config = load_parameters(self.get_parameter("robot").value, self.get_parameter("exp").value, "control")

        self.declare_parameter("controller_rate", self.control_config["nominal"]["frequency"])
        self.controller_rate = self.get_parameter("controller_rate").value
        # Initialize Controller

    def start_controller(self):

        # Initialize timer for controller
        self.create_timer(1 / self.controller_rate, self.publish_control)

    def callback_state(self, state_array_msg):
        """
        Callback for the state subscriber.

        Args:
            state_array_msg (Array): The incoming state message.

        This method updates the robot's state based on the incoming message.
        """
        self.process_state(np.array(state_array_msg.value))

    def process_state(self, state):
        """
        Process the new state.
        Base behavior: Update only state internally.
        """
        self.state = state

    def publish_control(self):
        """
        Publishes the prioritized control.
        """

        if self.state is None:
            if not getattr(self, "_warned_state", False):
                self.get_logger().warn("Waiting for state (odom) before publishing control…")
                self._warned_state = True
            return
        
        if self.controller_type == "HJR":
            prep = getattr(self, "controller_prep", None)
            tv_vf_ready = (prep is not None) and (getattr(prep, "tv_vf", None) is not None)
            if (getattr(self, "grid", None) is None) or (not tv_vf_ready):
                if not getattr(self, "_warned_vf", False):
                    self.get_logger().warn("Waiting for HJ value function/grid…")
                    self._warned_vf = True
                return


        # Get nominal control
        control = self.controller(self.state, self.get_clock().now().nanoseconds)  # Assuming controller is a callable
        control = control.squeeze()
        # self.get_logger().info(f"Control: {list(control)}")

        # Create control message
        control_msg = Array()
        control_msg.value = control.tolist()

        # Publish control message
        self.control_pub.publish(control_msg)

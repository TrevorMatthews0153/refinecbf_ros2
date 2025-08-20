#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from refinecbf_ros2.msg import Array
from refinecbf_ros2.srv import HighLevelCommand
from utils import load_parameters


class BaseInterface(Node):
    """
    BaseInterface is an abstract base class that converts the state and control messages
    from the SafetyFilterNode to the correct type for the crazyflies. Each hardware platform
    should have its own Interface node that subclasses this base class.

    Attributes:
    - state_msg_type: The ROS message type for the robot's state.
    - control_out_msg_type: The ROS message type for the robot's safe control.

    Subscribers:
    - ~topics/robot_state: Subscribes to the robot's state.
    - ~topics/cbf_safe_control: Subscribes to the safe control messages.

    Publishers:
    - state_pub (~topics/cbf_state): Publishes the converted state messages.
    - safe_control_pub (~topics/robot_safe_control): Publishes the converted safe control messages.
    """

    state_msg_type = None
    control_out_msg_type = None

    def __init__(self, node_name="base_interface"):
        super().__init__(node_name)
        # Get topics from parameters
        self.declare_parameters(
            "",
            [
                ("topics.robot_state", rclpy.Parameter.Type.STRING),
                ("topics.cbf_state", rclpy.Parameter.Type.STRING),
                ("topics.robot_safe_control", rclpy.Parameter.Type.STRING),
                ("topics.cbf_safe_control", rclpy.Parameter.Type.STRING),
                ("topics.robot_external_control", rclpy.Parameter.Type.STRING),
                ("topics.cbf_external_control", rclpy.Parameter.Type.STRING),
                ("services.highlevel_command", rclpy.Parameter.Type.STRING),
            ],
        )

        self.declare_parameter("robot", rclpy.Parameter.Type.STRING)
        self.declare_parameter("exp", rclpy.Parameter.Type.INTEGER)

        # Generate the update get parameters
        self.robot_state_topic = self.get_parameter("topics.robot_state").value
        cbf_state_topic = self.get_parameter("topics.cbf_state").value
        self.state_pub = self.create_publisher(Array, cbf_state_topic, 1)

        robot_safe_control_topic = self.get_parameter("topics.robot_safe_control").value
        self.cbf_safe_control_topic = self.get_parameter("topics.cbf_safe_control").value
        self.safe_control_pub = self.create_publisher(self.control_out_msg_type, robot_safe_control_topic, 1)

        self.robot_external_control_topic = self.get_parameter("topics.robot_external_control").value
        cbf_external_control_topic = self.get_parameter("topics.cbf_external_control").value
        self.external_control_pub = self.create_publisher(Array, cbf_external_control_topic, 1)

        high_level_command_srv = self.get_parameter("services.highlevel_command").value
        self.create_service(HighLevelCommand, high_level_command_srv, self.handle_high_level_command)

    def init_subscribers(self):
        self.create_subscription(self.state_msg_type, self.robot_state_topic, self.callback_state, 1)
        self.create_subscription(Array, self.cbf_safe_control_topic, self.callback_safe_control, 1)

    def callback_state(self, state_msg):
        """
        Callback for the state subscriber. This method should be implemented in a subclass.

        Args:
            state_msg: The incoming state message.
        """
        raise NotImplementedError("Must be subclassed")

    def handle_high_level_command(self, request, response):
        response.response = "actions not implemented (no impact)"
        return response

    def callback_safe_control(self, control_in_msg):
        """
        Callback for the safe control subscriber. This method should be implemented in a subclass.
        Should call self.override_safe_control()

        Args:
            control_msg: The incoming control message.
        """
        control_out_msg = self.process_safe_control(control_in_msg)
        assert isinstance(control_out_msg, self.control_out_msg_type), "Override to process the safe control message"
        if not self.override_safe_control():
            self.safe_control_pub.publish(control_out_msg)

    def process_safe_control(self, control_in_msg):
        raise NotImplementedError("Must be subclassed")

    def convert_and_clip_control_output(self, control_in_msg):
        return control_in_msg

    def override_safe_control(self):
        """
        Checks if the robot should override the safe control. Defaults to False.
        Should be overriden if the robot has to be able to be taken over by user.

        Returns:
            True if the robot should override the safe control, False otherwise.
        """
        return False

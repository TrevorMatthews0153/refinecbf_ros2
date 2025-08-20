#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import jax
import jax.numpy as jnp
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
from example_interfaces.msg import Bool
from refinecbf_ros2.msg import Array, ValueFunctionMsg, Obstacles
from refinecbf_ros2.srv import ActivateObstacle
import numpy as np
import hj_reachability as hj
import matplotlib.pyplot as plt
from config import Config


class ObstacleNode(Node):
    def __init__(self):
        # Following publishers:
        # - /env/obstacle_update
        # Following subscribers:
        # - /state
        super().__init__("obstacle_node")

        # Config:
        self.config = Config(self, hj_setup=True, obstacle_setup=True)
        self.dynamics = self.config.dynamics
        self.active_obstacles = self.config.active_obstacles
        self.active_obstacle_names = self.config.active_obstacle_names
        self.boundary = self.config.boundary
        self.safety_states_idis = self.config.safety_states
        self.robot_state = None
        # Parameter and Publisher setup:
        self.declare_parameter("vf_update_method", "pubsub")
        self.vf_update_method = self.get_parameter("vf_update_method").value
        self.declare_parameters(
            "",
            [
                ("topics.sdf_update", rclpy.Parameter.Type.STRING),
                ("topics.obstacle_update", rclpy.Parameter.Type.STRING),
                ("topics.cbf_state", rclpy.Parameter.Type.STRING),
            ],
        )
        sdf_update_topic = self.get_parameter("topics.sdf_update").value
        if self.vf_update_method == "pubsub":
            self.sdf_update_pub = self.create_publisher(ValueFunctionMsg, sdf_update_topic, 1)
        elif self.vf_update_method == "file":
            self.sdf_update_pub = self.create_publisher(Bool, sdf_update_topic, 1)
        else:
            raise NotImplementedError(f"{self.vf_update_method} is not a valid vf update method")

        obstacle_update_topic = self.get_parameter("topics.obstacle_update").value
        self.obstacle_update_pub = self.create_publisher(Obstacles, obstacle_update_topic, 1)

        # Subscribers:
        cbf_state_topic = self.get_parameter("topics.cbf_state").value
        self.create_subscription(Array, cbf_state_topic, self.callback_state, 1)

        # Initialize and Update Obstacles
        self.update_sdf()
        self.update_active_obstacles()

    def update_sdf(self):
        sdf = hj.utils.multivmap(self.build_sdf(), jnp.arange(self.config.grid.ndim))(self.config.grid.states) # * 10
        self.get_logger().info("Share Safe SDF {:.2f}".format(((sdf >= 0).sum() / sdf.size) * 100))
        if self.vf_update_method == "pubsub":
            self.sdf_update_pub.publish(ValueFunctionMsg(vf=sdf.flatten()))
        else:  # self.vf_update_method == "file"
            np.save("sdf.npy", sdf)
            self.sdf_update_pub.publish(Bool(data=True))

    def update_active_obstacles(self):
        self.obstacle_update_pub.publish(Obstacles(obstacle_names=self.active_obstacle_names))

    def callback_state(self, state_msg):
        self.robot_state = np.array(state_msg.value)[self.safety_states_idis]

    def build_sdf(self):
        def sdf(x):
            sdf_val = self.boundary.boundary_sdf(x)
            for obstacle in self.active_obstacles:
                obstacle_sdf = obstacle.obstacle_sdf(x)
                sdf_val = jnp.minimum(sdf_val, obstacle_sdf)
            return sdf_val

        return sdf


def main(args=None):
    rclpy.init(args=args)
    obstacle_node = ObstacleNode()

    rclpy.spin(obstacle_node)
    obstacle_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

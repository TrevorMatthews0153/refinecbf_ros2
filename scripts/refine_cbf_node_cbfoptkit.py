#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import jax
import jax.numpy as jnp
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
from refinecbf_ros2.msg import ValueFunctionMsg, Array, HiLoArray
from refinecbf_ros2.srv import ProcessState
from example_interfaces.msg import Bool, Float32
from cbf_opt_kit.safety_filter import ControlAffineSafetyFilter
from cbf_opt_kit.cbf import HJReachabilityControlAffineCBF
from config import Config
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import matplotlib.pyplot as plt
from utils import load_parameters
import threading
import hj_reachability as hj
from dataclasses import dataclass
import time



@dataclass
class HJModel:
    grid: hj.Grid
    grid_values: jnp.ndarray

class SafetyFilterNode(Node):
    """
    Docstring for the ROS2 version of the SafetyFilterNode
    """

    def __init__(self):
        super().__init__("safety_filter_node")
        self.initialized_safety_filter = False  # Initialization flag for the callback
        self.config = Config(self, hj_setup=True)
        self.dynamics = self.config.dynamics
        self.grid = self.config.grid
        self.safety_states_idis = self.config.safety_states
        self.safety_controls_idis = self.config.safety_controls
        self.vf_update_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_update_callback_group = MutuallyExclusiveCallbackGroup()
        self.other_callback_group = MutuallyExclusiveCallbackGroup()
        self.number_of_HJ_updates = 1
        # Parameters (for topics)
        self.declare_parameters(
            "",
            [
                ("topics.vf_update", rclpy.Parameter.Type.STRING),
                ("topics.cbf_state", rclpy.Parameter.Type.STRING),
                ("topics.cbf_nominal_control", rclpy.Parameter.Type.STRING),
                ("topics.cbf_safe_control", rclpy.Parameter.Type.STRING),
                ("topics.value_function", rclpy.Parameter.Type.STRING),
            ],
        )

        self.declare_parameter("safety_filter_active", True)
        self.declare_parameter("vf_update_method", "pubsub")

        control_config = load_parameters(self.get_parameter("robot").value, self.get_parameter("exp").value, "control")

        self.declare_parameter("control.asif.gamma", control_config["asif"]["gamma"])
        self.declare_parameter("control.asif.slack", control_config["asif"]["slack"])
        self.declare_parameter("control.asif.weighting", control_config["asif"].get("weighting"))

        self.declare_parameter("control.nominal.frequency", control_config["nominal"]["frequency"])
        # Subscribers
        self.vf_update_method = self.get_parameter("vf_update_method").value
        vf_topic = self.get_parameter("topics.vf_update").value
        if self.vf_update_method == "pubsub":
            self.vf_sub = self.create_subscription(ValueFunctionMsg, vf_topic, self.callback_vf_update_pubsub, 
                                                   1, callback_group=self.vf_update_callback_group)
        elif self.vf_update_method == "file":
            self.vf_sub = self.create_subscription(Bool, vf_topic, self.callback_vf_update_file, 
                                                   1, callback_group=self.vf_update_callback_group)
        else:
            raise NotImplementedError(f"{self.vf_update_method} is not a valid vf update method")

        state_topic = self.get_parameter("topics.cbf_state").value
        self.state_sub = self.create_subscription(Array, state_topic, self.callback_state, 
                                                  1, callback_group=self.other_callback_group)
        self.state = None

        # CBF setup
        gamma = self.get_parameter("control.asif.gamma").value
        weighting = self.get_parameter("control.asif.weighting").value
        # self.get_logger().info(f"Using weighting: {weighting}")
        # self.get_logger().info(f"Using weighting type: {type(weighting)}")
        if weighting is None:
            # self.get_logger().info('Weighting is None ...........................................')
            weighting = np.ones(len(self.safety_controls_idis))
        else:
            # self.get_logger().info('Weighting is NOT None ...........................................')
            weighting = np.array(weighting)
        self.get_logger().info(f"Using weighting: {weighting}")
        slackify_safety_constraint = self.get_parameter("control.asif.slack").value
        self.nominal_frequency = self.get_parameter("control.nominal.frequency").value
        self.nominal_time_period = 1.0 / self.nominal_frequency
        self.get_logger().info(f"Using gamma: {gamma}, slackify: {slackify_safety_constraint}")
        alpha = lambda x: gamma * x
        self.hj_model = HJModel(grid=self.grid, grid_values=jnp.zeros(self.config.grid_shape))
        self.hj_model_back = HJModel(grid=self.grid, grid_values=jnp.zeros(self.config.grid_shape))
        self.get_logger().info(f"control space: {self.dynamics.control_space}")
        self.active_buffer_cbf = HJReachabilityControlAffineCBF(self.dynamics, model=self.hj_model, time_invariant=True, logger=self.get_logger())
        self.back_buffer_cbf = HJReachabilityControlAffineCBF(self.dynamics, model=self.hj_model, time_invariant=True, logger=self.get_logger())
        self.lock = threading.Lock()

        backup_control = ControlAffineSafetyFilter(self.active_buffer_cbf, alpha=alpha,
                                                   weighting=weighting,
                                                #    constrain_controls=False,
                                                   return_values=True,
                                                   logger=self.get_logger())
        self.safety_filter_solver = ControlAffineSafetyFilter(
            self.active_buffer_cbf,
            alpha=alpha,
            weighting=weighting,
            backup_filter=backup_control,
            return_values=True,
            logger=self.get_logger()
        )

        # Control subscriptions
        nom_control_topic = self.get_parameter("topics.cbf_nominal_control").value
        self.nominal_control_sub = self.create_subscription(Array, nom_control_topic, self.callback_safety_filter, 
                                                            1, callback_group=self.control_update_callback_group)

        filtered_control_topic = self.get_parameter("topics.cbf_safe_control").value
        self.pub_filtered_control = self.create_publisher(Array, filtered_control_topic, 1, callback_group=self.control_update_callback_group)

        # Value function publishing
        value_function_topic = self.get_parameter("topics.value_function").value
        self.value_function_pub = self.create_publisher(Float32, value_function_topic, 1)
        self.state_associated_with_vf_pub = self.create_publisher(Array, '/debugging/state', 1)
        self.control_associated_with_vf_pub = self.create_publisher(Array, '/debugging/safe_control', 1)
        self.nominal_control_associated_with_vf_pub = self.create_publisher(Array, '/debugging/nominal_control', 1)

        self.safety_filter_active = self.get_parameter("safety_filter_active").value
        if self.safety_filter_active:
            self.initialized_safety_filter = False
            # ControlAffineSafetyFilter doesn't need setup_optimization_problem
            self.get_logger().info("Safety filter is used, but not initialized yet")
        else:  # FIXME: This is correct, should be put back in later (post debug)
            self.initialized_safety_filter = True
            self.safety_filter_solver = lambda state, time, nominal_control: (nominal_control, 0, 0, 0)
            self.get_logger().warn("No safety filter, be careful!")

    def callback_vf_update_file(self, vf_msg):
        if not vf_msg.data:
            return
        try:
            self.back_buffer_cbf.vf_table = np.load("/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp2/update_vf.npy").reshape(self.config.grid_shape)
        except (ValueError, EOFError):
            import time
            time.sleep(0.03)
            try:
                self.back_buffer_cbf.vf_table = np.load("/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp2/update_vf.npy").reshape(self.config.grid_shape)
            except EOFError:
                self.get_logger().warn("Value function file not found, skipping update")
                return
        with self.lock:
            self.swap_buffers()
        if not self.initialized_safety_filter:
            self.get_logger().info("Initialized safety filter")
            self.initialized_safety_filter = True

    def find_closest_safe_state_service(self, request, response):
        """
        Service to find closest safe state to requested target state
        """
        desired_state = jnp.array(request.desired_state.value)
        weighting = jnp.array(request.weighting.value)
        self.get_logger().info("Desired state: {}, weighting {}".format(desired_state, weighting))
        if (not self.safety_filter_active) or self.back_buffer_cbf.vf(desired_state, 0.0) >= 0.0:
            self.get_logger().warn("Current state is already safe")
            processed_state = Array()
            processed_state.value = desired_state.tolist()
            response.processed_state = processed_state
            return response

        with self.lock:
            contour = plt.contour(
                self.grid.coordinate_vectors[0],
                self.grid.coordinate_vectors[1],
                self.back_buffer_cbf.vf_table[:, :, self.grid.nearest_index(desired_state)[2], self.grid.nearest_index(desired_state)[3]].T,
                levels=[0.1],
            )
        array_points = [path.vertices for path in contour.collections[0].get_paths()][0]
        boundary_states = np.concatenate([array_points, np.zeros_like(array_points)], axis=1)
        self.get_logger().info("Boundary states: {}".format(boundary_states))
        closest_state = boundary_states[np.argmin(np.linalg.norm(weighting * (boundary_states- desired_state), axis=1))]
        processed_state = Array()
        processed_state.value = closest_state.tolist()
        response.processed_state = processed_state
        return response

    def swap_buffers(self):
        if self.active_buffer_cbf.vf_table is None:
            self.active_buffer_cbf._vf_table = self.back_buffer_cbf._vf_table.copy()
            self.active_buffer_cbf._grad_vf_table = self.back_buffer_cbf._grad_vf_table.copy()
        else:
            self.active_buffer_cbf, self.back_buffer_cbf = self.back_buffer_cbf, self.active_buffer_cbf
        self.safety_filter_solver.cbf = self.active_buffer_cbf

    def callback_vf_update_pubsub(self, vf_msg):
        self.get_logger().info(f"Updating VF backbuffer, iteration: {self.number_of_HJ_updates}")
        start_time = time.time()
        self.back_buffer_cbf.vf_table = np.array(vf_msg.vf).reshape(self.config.grid_shape) 
        self.back_buffer_cbf.vf_and_dv(np.ones(4), 0.0)  # Warmstart this
        end_time = time.time()
        self.get_logger().info(f"Time taken to update VF backbuffer: {end_time - start_time:.2f} seconds")
        with self.lock:
            self.get_logger().info(f"Swapping buffers, iteration: {self.number_of_HJ_updates}") 
            self.swap_buffers()
        if not self.initialized_safety_filter:
            self.get_logger().info("Initialized safety filter")
            self.initialized_safety_filter = True
        self.last_solved_time = time.time()
        self.number_of_HJ_updates += 1 

    def callback_safety_filter(self, control_msg):
        start_time = time.time()
        nom_control = np.array(control_msg.value)
        if self.state is None:
            self.get_logger().info("State not set yet, no control published", throttle_duration_sec=5.0)
            return
        if not self.initialized_safety_filter:
            safety_control_msg = Array()# = control_msg
            safety_control = nom_control.copy()
            safety_control[self.safety_controls_idis] = np.zeros(len(self.dynamics.CONTROLS))
            safety_control_msg.value = safety_control.tolist()
            self.get_logger().info("Safety filter not initialized yet, outputting zero control", throttle_duration_sec=2.0)
        else:
            nom_control_active = nom_control[self.safety_controls_idis]
            safety_control_msg = Array()
            curr_state = self.state.copy()
            safety_filter_tuple = self.safety_filter_solver(
                curr_state, time=0.0,nominal_control=np.array(nom_control_active)
            )

            if len(safety_filter_tuple) == 4 and self.safety_filter_active:
                self.value_function_pub.publish(Float32(data=safety_filter_tuple[3]))
                self.state_associated_with_vf_pub.publish(Array(value=curr_state.tolist()))
                self.control_associated_with_vf_pub.publish(Array(value=safety_filter_tuple[0].tolist()))
                self.nominal_control_associated_with_vf_pub.publish(Array(value=nom_control_active.tolist()))
            safety_control = nom_control.copy()
            safety_control[self.safety_controls_idis] = np.array(safety_filter_tuple[0])
            safety_control_msg.value = safety_control.tolist()

        self.pub_filtered_control.publish(safety_control_msg)
        if self.initialized_safety_filter:
            self.last_solved_sf = time.time()

    def callback_state(self, state_msg):
        self.state = np.array(state_msg.value)[self.safety_states_idis]


def main(args=None):
    rclpy.init(args=args)
    safety_filter = SafetyFilterNode()
    executor = MultiThreadedExecutor(num_threads=12)
    executor.add_node(safety_filter)
    try:
        executor.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

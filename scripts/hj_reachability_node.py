#!/usr/bin/env python3

import os
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node

import hj_reachability as hj
import jax
import jax.numpy as jnp

from std_msgs.msg import Bool as StdBool 
from std_msgs.msg import Float32
from refinecbf_ros2.msg import ValueFunctionMsg
from example_interfaces.msg import Bool

from utils import load_parameters, load_array
from config import Config, QuadraticCBF
from refine_cbfs import TabularControlAffineCBF


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
jax.config.update("jax_platform_name", "gpu")


class HJReachabilityNode(Node):
    """
    Computes/updates a value function using Hamilton–Jacobi reachability, with SDF updates coming
    via pubsub or a file-trigger topic.

    Publishes either:
      - ValueFunctionMsg (pubsub mode), OR
      - example_interfaces/Bool(True) and writes vf.npy (file mode)
    """

    def __init__(self) -> None:
        super().__init__("hj_reachability_node")

        # ---- Core config / dynamics
        self.config = Config(self, hj_setup=True)
        self.dynamics = self.config.dynamics
        self.grid = self.config.grid
        self.hj_dynamics = self.config.hj_dynamics
        self.control_space = self.hj_dynamics.control_space
        self.disturbance_space = self.hj_dynamics.disturbance_space

        # ---- Convergence
        self.convergence_iterations = 15
        self.convergence_count = 0

        # ---- Topic params
        self.declare_parameters(
            "",
            [
                ("topics.sdf_update", rclpy.Parameter.Type.STRING),
                ("topics.vf_update", rclpy.Parameter.Type.STRING),
                ("topics.safe_cell", rclpy.Parameter.Type.STRING),
            ],
        )

        # ---- Behavior params
        self.declare_parameter("vf_update_method", "file")
        self.declare_parameter("vf_update_accuracy", "very_high")
        self.declare_parameter("vf_initialization_method", "file")
        self.declare_parameter("initial_vf_file", "None")
        self.declare_parameter("update_vf_online", True)
        self.declare_parameter("do_hjr", True)
        self.declare_parameter("save_cbf", False)

        # ---- Read params
        self.vf_update_method = self.get_parameter("vf_update_method").value
        self.vf_update_accuracy = self.get_parameter("vf_update_accuracy").value
        self.vf_initialization_method = self.get_parameter("vf_initialization_method").value
        self.update_vf_flag = self.get_parameter("update_vf_online").value
        self.do_hjr = self.get_parameter("do_hjr").value
        self.save_cbf = self.get_parameter("save_cbf").value

        self.sdf_update_topic = self.get_parameter("topics.sdf_update").value
        self.vf_topic = self.get_parameter("topics.vf_update").value
        self.safe_cell_topic = self.get_parameter("topics.safe_cell").value

        self.sdf_available = False 

        self.get_logger().info(f"Save CBF after every goal: {self.save_cbf}")
        if not self.do_hjr:
            self.get_logger().info("HJ Reachability is disabled using SDF as value function!")
        if not self.update_vf_flag:
            self.get_logger().warn("Value function is not being updated")

        # ---- Load control config
        control_config = load_parameters(
            self.get_parameter("robot").value,
            self.get_parameter("exp").value,
            "control",
        )

        # ---- Subscriptions
        if self.vf_update_method == "pubsub":
            self.sdf_subscriber = self.create_subscription(
                ValueFunctionMsg, self.sdf_update_topic, self.callback_sdf_update_pubsub, 1
            )
        elif self.vf_update_method == "file":
            self.sdf_subscriber = self.create_subscription(
                Bool, self.sdf_update_topic, self.callback_sdf_update_file, 1
            )
        else:
            raise NotImplementedError(f"{self.vf_update_method} is not a valid vf update method")

        # ---- Optional goal saving
        self.current_goals_reached = 0
        if self.save_cbf:
            self.goal_reached_sub = self.create_subscription(Bool, "goal_reached", self.goal_reached_cb, 1)

        # ---- Wait for first SDF message using a background spin thread
        self.first_message_received = threading.Event()
        self.spin_thread = threading.Thread(target=self.spin)
        self.spin_thread.start()
        self.first_message_received.wait()

        # ---- Solver settings
        self.solver_settings = hj.SolverSettings.with_accuracy(self.vf_update_accuracy)

        # ---- VF initialization
        if self.vf_initialization_method == "sdf":
            self.get_logger().info("Initializing VF with SDF")
            self.vf = self.sdf_values.copy()

        elif self.vf_initialization_method == "cbf":
            cbf_params = control_config["initial_cbf"]
            self.get_logger().info("Initializing VF with CBF")
            original_cbf = QuadraticCBF(self.dynamics, cbf_params["Parameters"], test=False)
            tabular_cbf = TabularControlAffineCBF(self.dynamics, params={}, test=False, grid=self.grid)
            tabular_cbf.tabularize_cbf(original_cbf)
            self.vf = tabular_cbf.vf_table.copy()

        elif self.vf_initialization_method == "file":
            self.get_logger().info("Initializing VF with file")
            sdf_init = load_array(self.get_parameter("robot").value, self.get_parameter("exp").value, "sdf_sim")

            # stack to match grid shape if needed
            if len(sdf_init.shape) < len(self.config.grid_shape):
                for i in range(len(sdf_init.shape), len(self.config.grid_shape)):
                    sdf_init = jnp.stack([sdf_init] * self.config.grid_shape[i], axis=-1)

            unsafe_velocity = lambda x: 10 * (x[3] - 0.2)  # keep velocity above 0.2 m/s
            sdf_vel = hj.utils.multivmap(unsafe_velocity, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
            self.sdf_values = jnp.minimum(sdf_init, sdf_vel)
            self.vf = self.sdf_values.copy()

            if self.vf.ndim == self.grid.ndim + 1:
                self.vf = self.vf[-1]

            print("File Loaded")
            assert self.vf.shape == tuple(self.config.grid_shape), "vf file is not compatible with grid size"

        else:
            raise NotImplementedError(f"{self.vf_initialization_method} is not a valid initialization method")

        # ---- Publishers
        self.vf_pub = self._create_vf_publisher(self.vf_update_method, self.vf_topic)
        self.safe_cell_pub = self.create_publisher(Float32, self.safe_cell_topic, 1)

        # ---- Start
        self.publish_initial_vf()
        self.update_vf()  # spins indefinitely

    # -------------------------
    # Helpers
    # -------------------------

    def _create_vf_publisher(self, vf_update_method, topic):
        if vf_update_method == "pubsub":
            return self.create_publisher(ValueFunctionMsg, topic, 1)
        return self.create_publisher(Bool, topic, 1)

    def _publish_vf(self):
        if self.vf_update_method == "pubsub":
            self.vf_pub.publish(ValueFunctionMsg(vf=self.vf.flatten().tolist()))
        else:
            np.save("/root/ros2_ws//vf.npy", np.array(self.vf))
            self.vf_pub.publish(Bool(data=True))

    def spin(self):
        rclpy.spin(self)

    # -------------------------
    # Public methods
    # -------------------------

    def publish_initial_vf(self):
        self.get_logger().info("Number of subscribers: {}".format(self.vf_pub.get_subscription_count()))
        while self.vf_pub.get_subscription_count() < 1:
            self.get_logger().info("HJR node: Waiting for subscribers to connect")
            time.sleep(1)

        if self.vf_update_method == "pubsub":
            msg = ValueFunctionMsg()
            msg.vf = self.vf.flatten().tolist()
            if not self.update_vf_flag:
                self.vf_pub.publish(msg)
        else:
            np.save("/root/ros2_ws/vf.npy", self.vf.copy())
            if not self.update_vf_flag:
                self.vf_pub.publish(Bool(data=True))

    def goal_reached_cb(self, msg):
        if msg.data and self.save_cbf:
            self.get_logger().info("Goal reached, saving CBF and SDF")
            self.current_goals_reached += 1
            parent_dir = "/root/ros2_ws/noise_and_range_experiments/low_range_high_noise"
            np.save(f"{parent_dir}/goal_{self.current_goals_reached}_cbf.npy", np.array(self.vf))
            np.save(f"{parent_dir}/goal_{self.current_goals_reached}_sdf.npy", np.array(self.sdf_values))

    def callback_sdf_update_pubsub(self, msg):
        self.get_logger().info("SDF update received from pubsub")
        if not msg.vf:
            return

        sdf_init = jnp.array(msg.vf)
        if sdf_init.size != np.prod(self.config.grid_shape):
            try:
                sdf_init = sdf_init.reshape(self.config.grid_shape[0], self.config.grid_shape[1])
                for i in range(len(sdf_init.shape), len(self.config.grid_shape)):
                    sdf_init = jnp.stack([sdf_init] * self.config.grid_shape[i], axis=-1)
            except:
                return
        else:
            sdf_init = sdf_init.reshape(self.config.grid_shape)

        unsafe_velocity = lambda x: 10 * (x[3] - 0.2)  # keep velocity above 0.2 m/s
        sdf_vel = hj.utils.multivmap(unsafe_velocity, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        self.sdf_values = jnp.minimum(sdf_init, sdf_vel)

        if not self.first_message_received.is_set():
            self.first_message_received.set()
        else:
            self.solver_settings = hj.SolverSettings.with_accuracy(self.vf_update_accuracy)

    def callback_sdf_update_file(self, msg):
        if not msg.data:
            return

        try:
            sdf_init = jnp.array(load_array(self.get_parameter("robot").value, self.get_parameter("exp").value, "sdf_sim"))
        except:
            self.get_logger().error("SDF file not found or incorrect")
            return

        need = len(self.config.grid_shape) - sdf_init.ndim
        sdf_init = jnp.reshape(sdf_init, sdf_init.shape + (1,) * need)
        sdf_init = jnp.broadcast_to(sdf_init, self.config.grid_shape)

        unsafe_velocity = lambda x: 10 * (x[3] - 0.2)  # keep velocity above 0.2 m/s
        sdf_vel = hj.utils.multivmap(unsafe_velocity, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        self.sdf_values = jnp.minimum(sdf_init, sdf_vel)

        if not self.first_message_received.is_set():
            self.first_message_received.set()
        else:
            self.solver_settings = hj.SolverSettings.with_accuracy(self.vf_update_accuracy)

    def update_vf(self):
        while rclpy.ok():
            if self.do_hjr:
                if self.update_vf_flag:
                    time_start = time.time()
                    safe_cell_share = np.sum(self.vf >= 0) / self.vf.size
                    self.get_logger().info(f"Share of safe cells: {safe_cell_share:.3f}")
                    self.safe_cell_pub.publish(Float32(data=float(safe_cell_share)))

                    for i in range(5):
                        new_values = hj.step(
                            self.solver_settings,
                            self.hj_dynamics,
                            self.grid,
                            0.0,
                            self.vf,
                            -0.02,
                            progress_bar=False,
                        )
                        self.vf = jnp.minimum(new_values, self.sdf_values)
                        self.get_logger().info("Time taken: {:.2f} s".format(time.time() - time_start))
            else:
                self.vf = self.sdf_values.copy()
                safe_cell_share = np.sum(self.vf >= 0) / self.vf.size
                self.get_logger().info(f"Share of safe cells (no HJR): {safe_cell_share:.3f}")
                self.safe_cell_pub.publish(Float32(data=float(safe_cell_share)))
                time.sleep(1)

            if self.convergence_count < self.convergence_iterations and self.do_hjr:
                self.convergence_count += 1
            else:
                self._publish_vf()


def main(args=None):
    rclpy.init(args=args)
    hj_reachability_node = HJReachabilityNode()
    rclpy.spin(hj_reachability_node)
    hj_reachability_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

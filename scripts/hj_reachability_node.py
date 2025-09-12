#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import hj_reachability as hj
import jax.numpy as jnp
from refinecbf_ros2.msg import ValueFunctionMsg, HiLoArray
from std_srvs.srv import Trigger
import tqdm

# Ensure the following imports are compatible with ROS2 or appropriately adapted
from config import Config, QuadraticCBF, InitialCBF
from refine_cbfs import HJControlAffineDynamics, TabularControlAffineCBF
from example_interfaces.msg import Bool
from std_msgs.msg import Float32

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8" 
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" # Use 90% of GPU memory
import jax
jax.config.update("jax_platform_name", "gpu")
import threading
import time
from utils import load_parameters, load_array


class HJReachabilityNode(Node):
    """
    HJReachabilityNode is a ROS node that computes the Hamilton-Jacobi reachability for a robot.

    Subscribers:
    - disturbance_update_sub (~topics/disturbance_update): Updates the disturbance.
    - actuation_update_sub (~topics/actuation_update): Updates the actuation.
    - sdf_update_sub (~topics/sdf_update): Updates the obstacles.

    Publishers:
    - vf_pub (~topics/vf_update): Publishes the value function.
    """

    def __init__(self) -> None:
        """
        Initializes the HJReachabilityNode. It sets up ROS subscribers for disturbance, actuation, and obstacle updates,
        and a publisher for the value function. It also initializes the Hamilton-Jacobi dynamics and the value function.
        """
        # Load configuration
        super().__init__("hj_reachability_node")
        self.config = Config(self, hj_setup=True)
        # Initialize dynamics, grid, and Hamilton-Jacobi dynamics
        self.dynamics = self.config.dynamics
        self.grid = self.config.grid
        self.hj_dynamics = self.config.hj_dynamics
        self.control_space = self.hj_dynamics.control_space
        self.disturbance_space = self.hj_dynamics.disturbance_space
        self.convergence_count = 0

        # Topics (parameters)
        self.declare_parameters(
            "",
            [
                ("topics.sdf_update", rclpy.Parameter.Type.STRING),
                ("topics.vf_update", rclpy.Parameter.Type.STRING),
                ("topics.safe_cell", rclpy.Parameter.Type.STRING),
            ],
        )

        self.declare_parameter("vf_update_method", "file")
        self.declare_parameter("vf_update_accuracy", "very_high")
        self.declare_parameter("vf_initialization_method", "file")
        self.declare_parameter("initial_vf_file", "None")
        self.declare_parameter("update_vf_online", True)
        self.declare_parameter("do_hjr", True)
        
        self.service_to_start = False
        control_config = load_parameters(self.get_parameter("robot").value, self.get_parameter("exp").value, "control") #update goal

        self.vf_update_method = self.get_parameter("vf_update_method").value
        self.vf_update_accuracy = self.get_parameter("vf_update_accuracy").value
        # Initialize a lock for thread-safe value function updates
        # Get initial safe space and setup solver
        self.sdf_update_topic = self.get_parameter("topics.sdf_update").value
        self.sdf_available = False
        self.do_hjr = self.get_parameter("do_hjr").value


        if not self.do_hjr:
            self.get_logger().info("HJ Reachability is disabled using SDF as value function!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
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

        # Wait while not sdf update topic received  / Commented when you want to test with perfect SDF
        self.first_message_received = threading.Event()
        self.spin_thread = threading.Thread(target=self.spin)
        self.spin_thread.start()
        self.first_message_received.wait()

        #Test SDF Comment when working with file/pubsub 24August2025
        # Circle obstacles
        # safe_region_1 = lambda x: -1 * (0.5 - jnp.linalg.norm(x[:2] - jnp.array([-2.0, -2.0]))) # (radius - norm of (x,y))
        # safe_region_2 = lambda x: -1 * (0.5 - jnp.linalg.norm(x[:2] - jnp.array([2.0, 2.0]))) # (radius - norm of (x,y))
        # safe_region_3 = lambda x: -1 * (0.5 - jnp.linalg.norm(x[:2] - jnp.array([-2.0, 2.0]))) # (radius - norm of (x,y))
        # safe_region_4 = lambda x: -1 * (0.5 - jnp.linalg.norm(x[:2] - jnp.array([2.0, -2.0]))) # (radius - norm of (x,y))

        # sdf1 = hj.utils.multivmap(safe_region_1, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        # sdf2 = hj.utils.multivmap(safe_region_2, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        # sdf3 = hj.utils.multivmap(safe_region_3, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        # sdf4 = hj.utils.multivmap(safe_region_4, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        # self.sdf_circle_values = jnp.minimum(jnp.minimum(sdf1, sdf2), jnp.minimum(sdf3, sdf4))
        
        # Square obstacle
        # center1 = jnp.array([0.0, 0.0])
        # # center2 = jnp.array([2.0, 3.0])
        # a = 2.6
        # safe_boundary = lambda x: jnp.max(jnp.abs(x[:2] - center1)) - a  # square obstacle
        # # safe_region_2 = lambda x: jnp.max(jnp.abs(x[:2] - center2)) - a

        # Square boundary
        # center1 = jnp.array([0.0, 0.0])
        # a = 2.6  # half-side (side length = 2a)
        # boundary_outside_unsafe = lambda x: a - jnp.max(jnp.abs(x[:2] - center1))
        # sdf_boundary = hj.utils.multivmap(boundary_outside_unsafe, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        
        # unsafe_velocity = lambda x: 10 * (x[3] - 0.1)  # keep velocity above 0.1 m/s
        # # self.brt = lambda sdf_values: lambda t, x: jnp.minimum(x, sdf_values)
        # sdf_vel = hj.utils.multivmap(unsafe_velocity, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        # # self.sdf_values = jnp.minimum(jnp.minimum(sdf_boundary, sdf_vel), self.sdf_circle_values)
        # self.sdf_values = jnp.minimum(self.sdf_circle_values, sdf_vel)
        # self.sdf_values = jnp.minimum(jnp.minimum(sdf1, sdf2), sdf3)
        # print(self.sdf_values.shape)
        self.solver_settings = hj.SolverSettings.with_accuracy(self.vf_update_accuracy) #, value_postprocessor=self.brt(self.sdf_values))
        
        self.vf_initialization_method = self.get_parameter("vf_initialization_method").value
        if self.vf_initialization_method == "sdf":
            self.get_logger().info("Initializing VF with SDF")
            self.vf = self.sdf_values.copy()

        elif self.vf_initialization_method == "cbf":
            # Here the Quadratic CBF is based on obstacles we instead want to update it to use GP-SDF
            # cbf_params = control_config["initial_cbf"]
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
                # stack for each missing dimension
                for i in range(len(sdf_init.shape), len(self.config.grid_shape)):    
                    sdf_init = jnp.stack([sdf_init] * self.config.grid_shape[i], axis=-1)

            unsafe_velocity = lambda x: 10 * (x[3] - 0.1)  # keep velocity above 0.1 m/s
            sdf_vel = hj.utils.multivmap(unsafe_velocity, jnp.arange(self.config.grid.ndim))(self.config.grid.states) 
            self.sdf_values = jnp.minimum(sdf_init, sdf_vel)       
            # self.sdf_values = self.sdf_init
            self.vf = self.sdf_values.copy()

            # Use if update_vf_online is False
            # self.get_logger().info("Initializing VF with file")
            # self.sdf_values = load_array(self.get_parameter("robot").value, self.get_parameter("exp").value, "jdfsdjkfbs").reshape(self.config.grid_shape)
            # self.vf = self.sdf_values.copy()

            if self.vf.ndim == self.grid.ndim + 1:
                self.vf = self.vf[-1]
            print("File Loaded")
            assert self.vf.shape == tuple(self.config.grid_shape), "vf file is not compatible with grid size"

        else:
            raise NotImplementedError("{} is not a valid initialization method".format(self.vf_initialization_method))
        # self.get_logger().info(f"Share of safe cells: {np.sum(self.vf >= 0) / self.vf.size:.3f}")

        # Set up value function publisher
        self.vf_topic = self.get_parameter("topics.vf_update").value

        # Log a warning if the update flag is not set
        self.update_vf_flag = self.get_parameter("update_vf_online").value
        if not self.update_vf_flag:
            self.get_logger().warn("Value function is not being updated")

        # Publishers depending on the update method
        if self.vf_update_method == "pubsub":
            self.vf_pub = self.create_publisher(ValueFunctionMsg, self.vf_topic, 1)
        else:  # self.vf_update_method == "file"
            self.vf_pub = self.create_publisher(Bool, self.vf_topic, 1)

        self.safe_cell_pub = self.create_publisher(Float32, self.get_parameter("topics.safe_cell").value, 1)
        # Start updating the value function
        self.publish_initial_vf()
        self.update_vf()  # This method spins indefinitely

    def spin(self):
        rclpy.spin(self)

    def publish_initial_vf(self):
        # ROS2 uses a slightly different API for waiting for subscribers
        self.get_logger().info("Number of subscribers: {}".format(self.vf_pub.get_subscription_count()))
        while self.vf_pub.get_subscription_count() < 1: # was previously 2
            self.get_logger().info("HJR node: Waiting for subscribers to connect")
            time.sleep(1)
        if self.vf_update_method == "pubsub":
            msg = ValueFunctionMsg()
            msg.vf = self.vf.flatten().tolist()  # Ensure data is in a suitable format
            if not self.update_vf_flag:
                self.vf_pub.publish(msg)
        else:  # self.vf_update_method == "file"
            np.save("/root/ros2_ws/vf.npy", self.vf.copy())
            if not self.update_vf_flag:
                self.vf_pub.publish(Bool(data=True))  # Publish a Bool message indicating completion

    def callback_sdf_update_pubsub(self, msg):
        # need to think about message type (Pointcloud to vfmessage)
        """
        Callback for the obstacle update subscriber.

        Args:
            msg (ValueFunctionMsg): The incoming obstacle update message.

        This method updates the obstacle and the solver settings.
        """
        self.get_logger().info("SDF update received from pubsub")
        #self.sdf_values = jnp.array(msg.vf).reshape(self.config.grid_shape)
        if not msg.vf:
            return
        
        sdf_init = jnp.array(msg.vf)
        if sdf_init.size != np.prod(self.config.grid_shape):
            try:
                sdf_init = sdf_init.reshape(self.config.grid_shape[0], self.config.grid_shape[1])
                for i in range(len(sdf_init.shape), len(self.config.grid_shape)):
                    # stack for each missing dimension
                    sdf_init = jnp.stack([sdf_init] * self.config.grid_shape[i], axis=-1)
            except:
                # self.get_logger().error("SDF update has incorrect size")
                return
        else:
            sdf_init = sdf_init.reshape(self.config.grid_shape)

        unsafe_velocity = lambda x: 10 * (x[3] - 0.1)  # keep velocity above 0.1 m/s
        sdf_vel = hj.utils.multivmap(unsafe_velocity, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        self.sdf_values = jnp.minimum(sdf_init, sdf_vel)

        if not self.first_message_received.is_set():
            self.first_message_received.set()
        else:
            self.solver_settings = hj.SolverSettings.with_accuracy(self.vf_update_accuracy)

    def callback_sdf_update_file(self, msg):
        # self.get_logger().info("SDF update received from file trigger")

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
        unsafe_velocity = lambda x: 10 * (x[3] - 0.1)  # keep velocity above 0.1 m/s
        sdf_vel = hj.utils.multivmap(unsafe_velocity, jnp.arange(self.config.grid.ndim))(self.config.grid.states)
        self.sdf_values = jnp.minimum(sdf_init, sdf_vel)

        if not self.first_message_received.is_set():
            self.first_message_received.set()
        else:
            self.solver_settings = hj.SolverSettings.with_accuracy(self.vf_update_accuracy)

    def update_vf(self):
        """
        Continuously updates the value function and publishes it as long as the node is running and the update flag is set.
        """
        while rclpy.ok():
            if self.do_hjr:
                if self.update_vf_flag:
                    time_start = time.time()
                    safe_cell_share = np.sum(self.vf >= 0) / self.vf.size
                    # self.get_logger().info(f"Share of safe cells: {safe_cell_share:.3f}")
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
                need = len(self.config.grid_shape) - self.sdf_values.ndim
                self.vf = jnp.reshape(self.sdf_values, self.sdf_values.shape + (1,) * need)
                self.vf = jnp.broadcast_to(self.vf, self.config.grid_shape)

            if self.vf_update_method == "pubsub":
                if self.convergence_count < 15:
                    # self.get_logger().info("Convergence count: {}".format(self.convergence_count))
                    self.convergence_count += 1
                else:
                    self.vf_pub.publish(ValueFunctionMsg(vf=self.vf.flatten().tolist()))

            else:  # self.vf_update_method == "file"
                np.save("/root/ros2_ws//vf.npy", np.array(self.vf))
                if self.convergence_count < 15:
                    # self.get_logger().info("Convergence count: {}".format(self.convergence_count))
                    self.convergence_count += 1
                else:
                    self.vf_pub.publish(Bool(data=True))

            

    

def main(args=None):
    rclpy.init(args=args)
    hj_reachability_node = HJReachabilityNode()
    rclpy.spin(hj_reachability_node)
    hj_reachability_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

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
from ament_index_python.packages import get_package_share_directory
import yaml


# class TurtlebotNominalControl(NominalController):
#     def __init__(self):
#         self.declare_parameters(
#             "",
#             [
#                 ("controller_type", self.control_config["controller_type"]),
#             ],
#         )
#         self.controller_type = self.get_parameter("controller_type").value
#         super().__init__("tb_nominal_control", hj_setup = (self.controller_type == "HJR"))
#         self.max_vel = self.control_config["limits"]["max_vel"]
#         self.min_vel = self.control_config["limits"]["min_vel"]
#         self.max_omega = self.control_config["limits"]["max_omega"]
#         umin = np.array([self.min_vel, -self.max_omega])
#         umax = np.array([self.max_vel, self.max_omega])
#         self.target = np.array(self.control_config["nominal"]["goal"]["coordinates"])
#         self.controller_type = self.get_parameter("controller_type").value
#         if self.controller_type == "HJR":
#             self.umax_hjr = np.array(self.env_config["control_space"]["hi"])
#             self.umin_hjr = np.array(self.env_config["control_space"]["lo"])
#             self.padding = np.array(self.control_config["nominal"]["goal"]["padding"])
#             self.max_time = self.control_config["nominal"]["goal"]["max_time"]
#             self.time_intervals = self.control_config["nominal"]["goal"]["time_intervals"]
#             self.solver_accuracy = self.control_config["nominal"]["goal"]["solver_accuracy"]
#             assert self.solver_accuracy in ["low", "medium", "high", "very_high"]
#             self.hj_dynamics = self.config.hj_dynamics
#             self.grid = self.config.grid
#             self.controller_prep = NominalControlHJ(
#                 self.hj_dynamics,
#                 self.grid,
#                 final_time=self.max_time,
#                 time_intervals=self.time_intervals,
#                 solver_accuracy=self.solver_accuracy,
#                 target=self.target,
#                 padding=self.padding,
#             )
#             self.get_logger().info("Solving for nominal control, nominal control default is 0")
#             self.controller = lambda x, t: np.zeros(self.dynamics.control_dims)
#             self.controller = self.controller_prep.get_nominal_control

#         elif self.controller_type == "PD":
#             self.controller = NominalControlPD(target=self.target, umin=umin, umax=umax).get_nominal_control

#         else:
#             raise NotImplementedError(f"{self.controller_type} is not a valid controller type")

#         self.get_logger().info("Nominal controller ready!")




class TurtlebotNominalControl(NominalController):
    def __init__(self):
        # Call parent constructor early to get Node functionality
        super().__init__("tb_nominal_control")

        # --- Load config files from ROS parameters ---
        self.declare_parameter("control_config_file", "")
        self.declare_parameter("env_config_file", "")

        control_config_path = self.get_parameter("control_config_file").get_parameter_value().string_value
        env_config_path = self.get_parameter("env_config_file").get_parameter_value().string_value

        # Load control config
        if control_config_path and os.path.exists(control_config_path):
            with open(control_config_path, "r") as f:
                self.control_config = yaml.safe_load(f)
        else:
            self.get_logger().error(f"control_config_file not found: {control_config_path}")
            self.control_config = {}

        # Load env config
        if env_config_path and os.path.exists(env_config_path):
            with open(env_config_path, "r") as f:
                self.env_config = yaml.safe_load(f)
        else:
            self.get_logger().error(f"env_config_file not found: {env_config_path}")
            self.env_config = {}

        # --- Now we can declare parameters that depend on the configs ---
        self.declare_parameter("controller_type", self.control_config.get("controller_type", "PD"))
        self.controller_type = self.get_parameter("controller_type").value

        # --- Limits ---
        if self.controller_type == "PD_acc":
            self.max_acc = self.control_config["limits"]["max_acc"]
            self.min_acc = self.control_config["limits"]["min_acc"]
            self.max_omega = self.control_config["limits"]["max_omega"]
            umin = np.array([self.min_acc, -self.max_omega])
            umax = np.array([self.max_acc, self.max_omega])
        else:
            self.max_vel = self.control_config["limits"]["max_vel"]
            self.min_vel = self.control_config["limits"]["min_vel"]
            self.max_omega = self.control_config["limits"]["max_omega"]
            umin = np.array([self.min_vel, -self.max_omega])
            umax = np.array([self.max_vel, self.max_omega])


        # --- Goal ---
        self.target = np.array(self.control_config["nominal"]["goal"]["coordinates"])
        print(self.target)

        # --- Controller selection ---
        if self.controller_type == "HJR":
            self.umax_hjr = np.array(self.env_config["control_space"]["hi"])
            self.umin_hjr = np.array(self.env_config["control_space"]["lo"])
            self.padding = np.array(self.control_config["nominal"]["goal"]["padding"])
            self.max_time = self.control_config["nominal"]["goal"]["max_time"]
            self.time_intervals = self.control_config["nominal"]["goal"]["time_intervals"]
            self.solver_accuracy = self.control_config["nominal"]["goal"]["solver_accuracy"]
            assert self.solver_accuracy in ["low", "medium", "high", "very_high"]

            self.hj_dynamics = self.config.hj_dynamics
            self.grid = self.config.grid
            self.controller_prep = NominalControlHJ(
                self.hj_dynamics,
                self.grid,
                final_time=self.max_time,
                time_intervals=self.time_intervals,
                solver_accuracy=self.solver_accuracy,
                target=self.target,
                padding=self.padding,
                progress_bar = True
            )
            self.get_logger().info("Solving for nominal control...")
            self.controller = self.controller_prep.get_nominal_control

        elif self.controller_type == "PD":
            self.controller = NominalControlPD(target=self.target, umin=umin, umax=umax).get_nominal_control

        elif self.controller_type == "PD_acc":
            self.controller = NominalControlPDAcc(target=self.target, umin=umin, umax=umax).get_nominal_control

        else:
            raise NotImplementedError(f"{self.controller_type} is not a valid controller type")

        self.get_logger().info("Nominal controller ready!")
        self.start_controller()

def main(args=None):
    rclpy.init(args=args)
    controller = TurtlebotNominalControl()

    try:
        while rclpy.ok():
            rclpy.spin(controller)
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

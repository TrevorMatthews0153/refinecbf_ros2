import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

class NominalPolicy:

    def __init__(self, ctrl):
        self.ctrl = ctrl

    def __call__(self, x, t):
        return self.ctrl.get_nominal_control(x, t)

    def save_measurements(self, state, control, time):
        return {"dist_to_goal": np.linalg.norm(state[..., :2] - self.ctrl.target[:2], axis=-1)}

class NominalControlPDAcc:
    def __init__(self, **kwargs):
        self.target = kwargs.get("target")
        self.umin = kwargs.get("umin")
        self.umax = kwargs.get("umax")
        self.env_config = kwargs.get("env_config")
        self.max_vel_des = 1.0
        self.min_vel_des = 0.1
    def nominal_policy(self, x, t):
        Kp = 0.3
        Ka = 0.5
        Kw = 2
        dx = self.target[0] - x[0]
        dy = self.target[1] - x[1]
        theta = x[2]
        # angle to target
        angle_to_target = jnp.arctan2(dy, dx)
        # heading error
        angle_error = angle_to_target - theta
        angle_error = jnp.arctan2(jnp.sin(angle_error), jnp.cos(angle_error))
        desired_vel = jnp.clip(Kp * (jnp.linalg.norm(self.target[0:2] - x[0:2])), self.min_vel_des, self.max_vel_des)
        acc = jnp.clip(Ka * (desired_vel - x[3]), self.umin[1], self.umax[1])
        omega = jnp.clip(Kw * angle_error, self.umin[0], self.umax[0])
        return jnp.array([acc, omega])
    def get_nominal_control(self, x, t):
        return self.nominal_policy(x, t)
    def get_nominal_controller(self, target):
        return self.nominal_policy

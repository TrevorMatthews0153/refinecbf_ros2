import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


class NominalControlPD:
    def __init__(self, **kwargs):
        self.target = kwargs.get("target")
        self.umin = kwargs.get("umin")
        self.umax = kwargs.get("umax")
        Kp = 1
        Kw = 2
        self.nominal_policy = lambda x,t: np.clip(
                [[
                  Kp*(np.linalg.norm(self.target[0:2]-x[0:2])),
                #   0.0]],
            Kw*np.arctan2(np.cos(x[2])*-(x[0]-self.target[0])+np.sin(x[2])*-(x[1]-self.target[1]),-np.sin(x[2])*-(x[0]-self.target[0])+np.cos(x[2])*-(x[1]-self.target[1]))]], 
            self.umin, self.umax)
    
    def get_nominal_control(self, x, t):
        return self.nominal_policy(x,t)


    def get_nominal_controller(self, target):
        return self.nominal_policy

# class NominalControlPD:
#     def __init__(self, **kwargs):
#         self.target = kwargs.get("target")
#         self.umin = np.asarray(kwargs.get("umin"))  # [v_min, -w_max]
#         self.umax = np.asarray(kwargs.get("umax"))  # [v_max,  w_max]
#         self.Kp = 1.0
#         self.Kw = 2.0
#         self.stop_radius = 0.05  # m, stop near goal
#         self.heading_slowdown = True  # reduce v when heading error is large

#         def _policy(x, t):
#             # x = [x, y, theta]
#             dx = self.target[0] - x[0]
#             dy = self.target[1] - x[1]
#             dist = np.hypot(dx, dy)

#             # Heading error (world to robot)
#             desired_yaw = np.arctan2(dy, dx)
#             e_th = desired_yaw - x[2]
#             # wrap to [-pi, pi]
#             e_th = (e_th + np.pi) % (2*np.pi) - np.pi

#             # PD-style commands
#             v = self.Kp * dist
#             if self.heading_slowdown:
#                 # Don’t charge forward if we’re not facing the target
#                 v *= max(0.0, np.cos(e_th))

#             w = self.Kw * e_th

#             # Stop near the target
#             if dist <= self.stop_radius:
#                 v = 0.0
#                 w = 0.0

#             u = np.array([v, w], dtype=np.float32)  # ORDER: [v, w]
#             u = np.clip(u, self.umin, self.umax)    # clip with matching bounds
#             return u

#         self.nominal_policy = _policy
    
#     def get_nominal_control(self, x, t):
#         return self.nominal_policy(x, t)

#     def get_nominal_controller(self, target):
#         return self.nominal_policy



class NominalPolicy:

    def __init__(self, ctrl):
        self.ctrl = ctrl

    def __call__(self, x, t):
        return self.ctrl.get_nominal_control(x, t)

    def save_measurements(self, state, control, time):
        return {"dist_to_goal": np.linalg.norm(state[..., :2] - self.ctrl.target[:2], axis=-1)}

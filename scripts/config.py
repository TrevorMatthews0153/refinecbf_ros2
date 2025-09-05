import hj_reachability as hj
import jax.numpy as jnp
import jax
from cbf_opt import ControlAffineDynamics, ControlAffineCBF
from refine_cbfs import HJControlAffineDynamics
import numpy as np
import rclpy
from utils import load_parameters
from dataclasses import dataclass


class Config:
    def __init__(self, node, hj_setup=False, obstacle_setup=False):
        # if param robot does not exist declare it
        node.declare_parameter('robot', rclpy.Parameter.Type.STRING)
        node.declare_parameter("exp", rclpy.Parameter.Type.INTEGER)  # Either str or int FIXME: only int supported
        robot = node.get_parameter('robot').value
        exp = node.get_parameter('exp').value
        # rclpy get path to package
        
        config = load_parameters(robot, exp, 'env')
        node.env_config = config
        # Load from config file (yaml)
        self.dynamics_class = config["dynamics_class"]
        self.dynamics = self.setup_dynamics()
        self.control_space = config["control_space"]
        self.dynamics.set_control_space(self.control_space)
        self.disturbance_space = config["disturbance_space"]
        self.dynamics.set_disturbance_space(self.disturbance_space)
        self.safety_states = config["safety_states"]
        self.safety_controls = config["safety_controls"]
        self.state_domain = config["state_domain"]
        self.grid_shape = np.array(self.state_domain["resolution"])

        self.obstacle_list = config.get("obstacles", [])
        self.boundary_env = config["boundary"]        

        if hj_setup:
            self.grid = self.setup_grid()  # This 
            if self.control_space["n_dims"] == 0:
                control_space_hj = hj.sets.Box(
                    lo=jnp.array([]), hi=jnp.array([]))
            else:
                control_space_hj = hj.sets.Box(
                    lo=jnp.array(self.control_space["lo"]), hi=jnp.array(self.control_space["hi"])
                )
            if self.disturbance_space["n_dims"] == 0:
                dist_space_hj = hj.sets.Box(lo=jnp.array([]), hi=jnp.array([]))
            else:
                dist_space_hj = hj.sets.Box(
                    lo=jnp.array(self.disturbance_space["lo"]), hi=jnp.array(self.disturbance_space["hi"])
                )
            self.hj_dynamics = HJControlAffineDynamics(
                self.dynamics, control_space=control_space_hj, disturbance_space=dist_space_hj
            )

        # self.assert_valid(hj_setup)

        if obstacle_setup:
            (
                self.active_obstacles,
                self.active_obstacle_names,
                self.boundary,
            ) = self.setup_obstacles()

    def assert_valid(self, hj_setup):
        assert len(self.control_space["lo"]) == self.dynamics.control_dims
        assert len(self.control_space["hi"]) == self.dynamics.control_dims
        assert self.dynamics.n_dims == len(self.state_domain["resolution"])

        if hj_setup:
            if len(self.obstacle_list) != 0:
                for obstacle in self.obstacle_list.values():
                    if obstacle["type"] == "Circle":
                        assert len(obstacle["center"]) == len(
                            obstacle["indices"])
                    if obstacle["type"] == "Rectangle":
                        assert len(obstacle["minVal"]) == len(
                            obstacle["indices"])
                        assert len(obstacle["maxVal"]) == len(
                            obstacle["indices"])
            assert len(self.boundary_env["minVal"]) == len(
                self.boundary_env["indices"])
            assert len(self.boundary_env["maxVal"]) == len(
                self.boundary_env["indices"])
    
    def setup_obstacles(self):
        # Obstacles that are "detected" by the robot when in close enough range
        active_obstacles = []  # Obstacles that are always active
        active_obstacle_names = []  # Names of the active Obstacles
        if len(self.obstacle_list) != 0:
            for name, obstacle in self.obstacle_list.items():      
                active_obstacle_names.append(name)
                if obstacle["type"] == "Circle":
                    active_obstacles.append(
                        Circle(
                            stateIndices=obstacle["indices"],
                            obstacleName=name,
                            radius=obstacle["radius"],
                            center=obstacle["center"],
                            padding=obstacle["padding"],
                        )
                    )
                elif obstacle["type"] == "Rectangle":
                    active_obstacles.append(
                        Rectangle(
                            stateIndices=obstacle["indices"],
                            obstacleName=name,
                            minVal=obstacle["minVal"],
                            maxVal=obstacle["maxVal"],
                            padding=obstacle["padding"],
                        )
                    )

        boundary = Boundary(
            stateIndices=self.boundary_env["indices"],
            minVal=self.boundary_env["minVal"],
            maxVal=self.boundary_env["maxVal"],
            padding=self.boundary_env["padding"],
        )

        return active_obstacles, active_obstacle_names, boundary

    def setup_dynamics(self):
        if self.dynamics_class == "quad_near_hover":
            return QuadNearHoverPlanarDynamics(params={"g": 9.81}, dt=0.05, test=False)
        elif self.dynamics_class == "dubins_car":
            return DubinsCarDynamics(params={"g": 9.81}, dt=0.05, test=False)
        elif self.dynamics_class == "dubins_acceleration":
            return DubinsAccelerationDynamics(params={"g": 9.81}, dt=0.05, test=False)
        else:
            raise ValueError(
                "Invalid dynamics type: {}".format(self.dynamics_class))

    def setup_grid(self):
        bounding_box = hj.sets.Box(lo=jnp.array(
            self.state_domain["lo"]), hi=jnp.array(self.state_domain["hi"]))
        grid_resolution = self.state_domain["resolution"]
        p_dims = self.state_domain["periodic_dims"]
        return hj.Grid.from_lattice_parameters_and_boundary_conditions(
            bounding_box, grid_resolution, periodic_dims=p_dims
        )


# Dynamics Classes
class QuadNearHoverPlanarDynamics(ControlAffineDynamics):
    """
    Simplified dynamics, and we need to convert controls from phi to tan(phi)"""

    STATES = ["y", "z", "v_y", "v_z"]
    CONTROLS = ["tan(phi)", "T"]
    DISTURBANCES = ["dy", "dvy"]

    def open_loop_dynamics(self, state, time: float = 0.0):
        return jnp.array([state[2], state[3], 0.0, -self.params["g"]])

    def control_matrix(self, state, time: float = 0.0):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [-self.params["g"], 0.0], [0.0, 1.0]])

    def disturbance_matrix(self, state, time: float = 0.0):
        return jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

@dataclass
class ControlSpace:
    control_dim: int
    lo: jnp.ndarray
    hi: jnp.ndarray

class DubinsCarDynamics(ControlAffineDynamics):
    """
    Dubins Car Dynamics for the Turtlebot
    """

    STATES = ["x", "y", "theta"]
    CONTROLS = ["v", "omega"]
    # DISTURBANCES = ["dx", "dy"]

    def open_loop_dynamics(self, state, time: float = 0):
        return jnp.array([0.0, 0.0, 0.0]) # maybe (vcos(theta), vsin(theta), 0.0) ?

    def control_matrix(self, state, time: float = 0.0):
        return jnp.array([[jnp.cos(state[2]), 0.0], [jnp.sin(state[2]), 0.0], [0.0, 1.0]])

    # def disturbance_jacobian(self, state, time: float = 0.0):
    #     return jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    def set_control_space(self, control_space):
        self.control_space = ControlSpace(control_dim = control_space['n_dims'],lo = jnp.array(control_space['lo']), hi = jnp.array(control_space['hi']))

class DubinsAccelerationDynamics(ControlAffineDynamics):
    """
    Dubins Car Dynamics w/ Acceleration
    """

    STATES = ["x", "y", "theta", "v"]
    CONTROLS = ["a", "omega"]
    DISTURBANCES = ["dx", "dy"]

    def open_loop_dynamics(self, state, time: float = 0):
        return jnp.array([state[3]*jnp.cos(state[2]),state[3]*jnp.sin(state[2]), 0.0, 0.0]) # maybe (vcos(theta), vsin(theta), 0.0) ?

    def control_matrix(self, state, time: float = 0.0):
        return jnp.array([[0.0, 0.0],
                         [0.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 0.0]])

    def set_control_space(self, control_space):
        self.control_space = ControlSpace(control_dim = control_space['n_dims'],lo = jnp.array(control_space['lo']), hi = jnp.array(control_space['hi']))

    def disturbance_matrix(self, state, time: float = 0.0):
        return jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    def set_disturbance_space(self, disturbance_space):
        self.disturbance_space = ControlSpace(control_dim = disturbance_space['n_dims'],lo = jnp.array(disturbance_space['lo']), hi = jnp.array(disturbance_space['hi']))


# Defining the dynamics of the quadrotor
class CrazyflieDynamics(ControlAffineDynamics):
    """
    Simplified dynamics, and we need to convert controls from phi to tan(phi)"""

    STATES = ["y", "z", "v_y", "v_z"]
    CONTROLS = ["tan(phi)", "T"]
    DISTURBANCES = []

    def __init__(self, params, test=True, **kwargs):
        super().__init__(params, test, **kwargs)

    def open_loop_dynamics(self, state, time: float = 0.0):
        return jnp.array([state[2], state[3], 0.0, -self.params["g"]])

    def control_matrix(self, state, time: float = 0.0):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [self.params["g"], 0.0], [0.0, 1.0]])

    def state_jacobian(self, state, control, disturbance=None, time: float = 0.0):
        return jax.jacfwd(lambda x: self.__call__(x, control, disturbance, time))(state)


# Implementing creating CBF
class QuadraticCBF(ControlAffineCBF):
    def __init__(self, dynamics, params, test=False, **kwargs):
        self.scaling = params["scaling"]
        self.center = params["center"]
        self.offset = params["offset"]
        self._vf_grad = jax.vmap(
            jax.grad(self.vf, argnums=0), in_axes=(0, None))
        super().__init__(dynamics, params=params, test=False, **kwargs)

    def vf(self, state, time=0.0):
        val = self.offset - \
            jnp.sum(np.array(self.scaling) *
                    (state - np.array(self.center)) ** 2, axis=-1)
        return val

    def _grad_vf(self, state, time=0.0):
        return self._vf_grad(state, time)

class InitialCBF(ControlAffineCBF):
    def __init__(self, dynamics, grid_axes, psi_values, grad_x=None, grad_y=None, **kwargs):
        self.psi_interp = RegularGridInterpolator(grid_axes, psi_values, bounds_error=False, fill_value=None)
        self.grad_x_interp = (
            RegularGridInterpolator(grid_axes, grad_x, bounds_error=False, fill_value=None) if grad_x is not None else None
        )
        self.grad_y_interp = (
            RegularGridInterpolator(grid_axes, grad_y, bounds_error=False, fill_value=None) if grad_y is not None else None
        )
        super().__init__(dynamics, {}, **kwargs)

    def vf(self, state, time=0.0):
        query = np.atleast_2d(state)[..., :2]
        return self.psi_interp(query)
    
    def _grad_vf(self, state, time=0.0):
        state = np.atleast_2d(state)       # ensure (1,3)
        query = state[..., :2]
        grad = np.zeros((state.shape[0], 3))  # ensure (1,3)
        grad[:, 0] = self.grad_x_interp(query)
        grad[:, 1] = self.grad_y_interp(query)
        return grad


# Obstacle Classes
class Obstacle:
    def __init__(self, type, stateIndices, obstacleName, padding) -> None:
        self.type = type
        self.stateIndices = stateIndices
        self.obstacleName = obstacleName
        self.padding = padding


class Circle(Obstacle):
    def __init__(
        self, stateIndices, obstacleName, radius, center, padding=0
    ) -> None:
        super().__init__("Circle", stateIndices, obstacleName, padding)
        self.radius = radius
        self.center = jnp.reshape(np.array(center), (-1, 1))

    def obstacle_sdf(self, x):
        obstacle_sdf = (
            jnp.linalg.norm(
                jnp.array([self.center - jnp.reshape(x[..., self.stateIndices], (-1, 1))]))
            - self.radius
            - self.padding
        )
        return obstacle_sdf

    def distance_to_obstacle(self, state):
        point = state[self.stateIndices].reshape(-1)
        distance = np.linalg.norm(self.center.reshape(-1) - point) - \
            self.radius - self.padding
        return distance


class Ellipse(Obstacle):
    def __init__(
        self, stateIndices, obstacleName, axes, center, padding=0
    ) -> None:
        super().__init__("Ellipse", stateIndices, obstacleName, padding)
        self.axes = jnp.array(axes)  # (semi_major_axis, semi_minor_axis)
        self.center = jnp.reshape(jnp.array(center), (-1, 1))

    def obstacle_sdf(self, x):
        # Translate points to ellipse coordinate system
        points = jnp.reshape(x[..., self.stateIndices], (-1, 2)) - self.center.T

        # Ellipse parameters
        a, b = self.axes

        # Compute the absolute values of the translated point coordinates
        x_abs = jnp.abs(points[..., 0])
        y_abs = jnp.abs(points[..., 1])

        # Normalize the coordinates by the ellipse axes
        q = jnp.stack([x_abs / a, y_abs / b], axis=-1)
        w = q - q**3 / jnp.linalg.norm(q**3, axis=-1, keepdims=True)

        # Calculate the distances
        distance_outside = jnp.linalg.norm(w * jnp.array([a, b]), axis=-1) - 1
        distance_inside = jnp.min(jnp.abs(q), axis=-1) * jnp.linalg.norm(jnp.array([a, b])) - 1

        # Conditional selection for inside or outside
        obstacle_sdf = jax.lax.cond(
            jnp.any(q > 1.0, axis=-1),
            lambda _: distance_outside,
            lambda _: -distance_inside,
            operand=None
        ) - self.padding

        # Reshape to the desired output shape
        output_shape = x.shape[:-1] + (201,)
        obstacle_sdf = jnp.reshape(obstacle_sdf, output_shape)
        return obstacle_sdf

    def distance_to_obstacle(self, state):
        # Translate points to ellipse coordinate system
        points = state[self.stateIndices].reshape(-1, 2) - self.center.T

        # Ellipse parameters
        a, b = self.axes

        # Compute the absolute values of the translated point coordinates
        x_abs = np.abs(points[:, 0])
        y_abs = np.abs(points[:, 1])

        # Normalize the coordinates by the ellipse axes
        q = np.stack([x_abs / a, y_abs / b], axis=-1)
        w = q - q**3 / np.linalg.norm(q**3, axis=-1, keepdims=True)

        # Calculate the distances
        distance_outside = np.linalg.norm(w * np.array([a, b]), axis=-1) - 1
        distance_inside = np.min(np.abs(q), axis=-1) * np.linalg.norm(np.array([a, b])) - 1

        distance = np.where(np.any(q > 1.0, axis=-1), distance_outside, -distance_inside)
        return distance - self.padding


class Rectangle(Obstacle):
    def __init__(
        self, stateIndices, obstacleName, minVal, maxVal, padding=0
    ) -> None:
        super().__init__("Rectangle", stateIndices, obstacleName, padding)
        self.minVal = jnp.reshape(np.array(minVal), (-1, 1))
        self.maxVal = jnp.reshape(np.array(maxVal), (-1, 1))

    def obstacle_sdf(self, x):
        max_dist_per_dim = jnp.max(
            jnp.array(
                [
                    self.minVal -
                    jnp.reshape(x[..., self.stateIndices], (-1, 1)),
                    jnp.reshape(x[..., self.stateIndices],
                                (-1, 1)) - self.maxVal,
                ]
            ),
            axis=0,
        )

        def outside_obstacle(_):
            return jnp.linalg.norm(jnp.maximum(max_dist_per_dim, 0))

        def inside_obstacle(_):
            return jnp.max(max_dist_per_dim)

        obstacle_sdf = (
            jax.lax.cond(jnp.all(max_dist_per_dim < 0.0),
                         inside_obstacle, outside_obstacle, operand=None)
            - self.padding
        )
        return obstacle_sdf

    def distance_to_obstacle(self, state):
        point = state[self.stateIndices].reshape(-1)
        minVal = self.minVal.reshape(-1)
        maxVal = self.maxVal.reshape(-1)
        max_dist_per_dim = np.max(np.array([minVal - point, point - maxVal]), axis=0)
        raw_distance = np.where(np.all(max_dist_per_dim < 0.0), 
                                np.max(max_dist_per_dim), 
                                np.linalg.norm(np.maximum(0, max_dist_per_dim)))
        return raw_distance - self.padding


class Boundary(Obstacle):
    def __init__(self, stateIndices, minVal, maxVal, padding=0) -> None:
        super().__init__("Boundary", stateIndices, None, padding)
        self.minVal = jnp.reshape(np.array(minVal), (-1, 1))
        self.maxVal = jnp.reshape(np.array(maxVal), (-1, 1))

    def boundary_sdf(self, x):
        max_dist_per_dim = jnp.max(
            jnp.array(
                [
                    self.minVal -
                    jnp.reshape(x[..., self.stateIndices], (-1, 1)),
                    jnp.reshape(x[..., self.stateIndices],
                                (-1, 1)) - self.maxVal,
                ]
            ),
            axis=0,
        )

        def outside_boundary(_):
            return -jnp.linalg.norm(jnp.maximum(max_dist_per_dim, 0))

        def inside_boundary(_):
            return -jnp.max(max_dist_per_dim)

        obstacle_sdf = (
            jax.lax.cond(jnp.all(max_dist_per_dim < 0.0),
                         inside_boundary, outside_boundary, operand=None)
            - self.padding
        )
        return obstacle_sdf


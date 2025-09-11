#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from example_interfaces.msg import Bool
from refinecbf_ros2.msg import ValueFunctionMsg

from geometry_msgs.msg import Vector3
from erl_gp_sdf_msgs.srv import SdfQuery


# -------------------------
# Helpers
# -------------------------

def parse_env_yaml(path):
    """
    Expect:
      state_domain:
        lo: [x_min, y_min, ...]
        hi: [x_max, y_max, ...]
        resolution: [nx, ny, ...]
        periodic_dims: [indices]  # unused here
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "state_domain" not in cfg:
        raise ValueError("YAML missing 'state_domain'")

    dom = cfg["state_domain"]
    lo = np.array(dom["lo"], dtype=np.float32)
    hi = np.array(dom["hi"], dtype=np.float32)
    res = np.array(dom["resolution"], dtype=np.int32)

    if not (len(lo) == len(hi) == len(res)):
        raise ValueError("lo, hi, resolution lengths must match")

    axes = [np.linspace(l, h, int(n), dtype=np.float32) for l, h, n in zip(lo, hi, res)]
    return {"lo": lo, "hi": hi, "resolution": res, "axes": axes}


def flatten_grid_rowmajor(ys: np.ndarray, xs: np.ndarray, z: float):
    """
    Create (Ny*Nx) Vector3 query points in row-major order:
    outer loop over y, inner loop over x.
    This means reshape(response, (Ny, Nx)) will align with (y,x).
    """
    pts = []
    for y in ys:
        for x in xs:
            v = Vector3()
            v.x = float(x)
            v.y = float(y)
            v.z = float(z)
            pts.append(v)
    return pts


# -------------------------
# Node
# -------------------------

class SDFServiceToGridNode(Node):
    """
    Queries SDF GP via erl_gp_sdf_msgs/SdfQuery and publishes:
      - ValueFunctionMsg on output_topic:     phi = sdf - var_sdf  (shape Ny x Nx, row-major → flattened col-major with .T like your node)
      - ValueFunctionMsg on output_topic_grad_x: d/dx(phi)  (uses gradient_x if provided; else np.gradient fallback)
      - ValueFunctionMsg on output_topic_grad_y: d/dy(phi)
    Also publishes one JSON String with grid metadata (same keys as your pointcloud node).
    Supports mode: 'pubsub' or 'file' (saving .npy and toggling Bool when saved).
    """

    def __init__(self):
        super().__init__("sdf_service_to_grid")

        # ---- Parameters (mirrors your current node where sensible)
        self.declare_parameter("env_config_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/env.yaml")
        self.declare_parameter("service_name", "sdf_query")
        self.declare_parameter("z", 0.0)                         # z plane for the query
        self.declare_parameter("publish_rate_hz", 10.0)           # query cadence
        self.declare_parameter("mode", "pubsub")                 # 'pubsub' or 'file'
        self.declare_parameter("output_topic", "/env/sdf_update")
        self.declare_parameter("output_topic_grad_x", "/env/sdf_grad_x_update")
        self.declare_parameter("output_topic_grad_y", "/env/sdf_grad_y_update")
        self.declare_parameter("grid_info_topic", "topics/sdf_grid_info")
        self.declare_parameter("sdf_file_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/sdf_sim.npy")
        self.declare_parameter("grad_x_file_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/curr_grad_x.npy")
        self.declare_parameter("grad_y_file_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/curr_grad_y.npy")
        self.declare_parameter("use_fallback_gradient_when_missing", False)

        # ---- Paths
        self.sdf_path = self.get_parameter("sdf_file_path").get_parameter_value().string_value
        self.gx_path  = self.get_parameter("grad_x_file_path").get_parameter_value().string_value
        self.gy_path  = self.get_parameter("grad_y_file_path").get_parameter_value().string_value

        # ---- Load env grid (x,y only)
        env_path = self.get_parameter("env_config_path").get_parameter_value().string_value
        dom = parse_env_yaml(env_path)
        self.xs = dom["axes"][0]
        self.ys = dom["axes"][1]
        self.Nx = int(self.xs.size)
        self.Ny = int(self.ys.size)
        self.z = float(self.get_parameter("z").value)

        # ---- Mode & pubs
        self.mode = self.get_parameter("mode").get_parameter_value().string_value.lower()
        out_vf_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        out_gx_topic = self.get_parameter("output_topic_grad_x").get_parameter_value().string_value
        out_gy_topic = self.get_parameter("output_topic_grad_y").get_parameter_value().string_value
        info_topic   = self.get_parameter("grid_info_topic").get_parameter_value().string_value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        if self.mode == "pubsub":
            self.vf_pub = self.create_publisher(ValueFunctionMsg, out_vf_topic, qos)
            self.gx_pub = self.create_publisher(ValueFunctionMsg, out_gx_topic, qos)
            self.gy_pub = self.create_publisher(ValueFunctionMsg, out_gy_topic, qos)
            self.info_pub = self.create_publisher(String, info_topic, qos)
        elif self.mode == "file":
            # publish Bool(True) per-channel when files are written
            self.vf_pub = self.create_publisher(Bool, out_vf_topic, qos)
            self.gx_pub = self.create_publisher(Bool, out_gx_topic, qos)
            self.gy_pub = self.create_publisher(Bool, out_gy_topic, qos)
            self.info_pub = None
        else:
            self.get_logger().warn(f"Unknown mode '{self.mode}', defaulting to 'pubsub'")
            self.mode = "pubsub"
            self.vf_pub = self.create_publisher(ValueFunctionMsg, out_vf_topic, qos)
            self.gx_pub = self.create_publisher(ValueFunctionMsg, out_gx_topic, qos)
            self.gy_pub = self.create_publisher(ValueFunctionMsg, out_gy_topic, qos)
            self.info_pub = self.create_publisher(String, info_topic, qos)

        self.first_info_sent = False
        self.use_fallback_grad = bool(self.get_parameter("use_fallback_gradient_when_missing").value)

        # ---- Service client
        self.service_name = self.get_parameter("service_name").get_parameter_value().string_value
        self.client = self.create_client(SdfQuery, self.service_name)

        # Prebuild query points (row-major y,x)
        self.query_pts = flatten_grid_rowmajor(self.ys, self.xs, self.z)

        # ---- Timer for periodic queries
        hz = float(self.get_parameter("publish_rate_hz").value)
        self.period = 1.0 / max(1e-6, hz)
        self.pending = False
        self.timer = self.create_timer(self.period, self._tick)

        self.get_logger().info(
            f"Ready. Mode={self.mode} service='{self.service_name}' grid=({self.Ny}x{self.Nx}) "
            f"x∈[{self.xs[0]:.2f},{self.xs[-1]:.2f}] y∈[{self.ys[0]:.2f},{self.ys[-1]:.2f}] z={self.z:.2f}"
        )

    # --------- Internals ----------

    def _tick(self):
        if self.pending:
            return

        if not self.client.service_is_ready():
            now = self.get_clock().now()
            if (not hasattr(self, "_last_warn_time") or
                (now - self._last_warn_time).nanoseconds * 1e-9 > 5.0):
                self.get_logger().warn(f"Service '{self.service_name}' not ready")
                self._last_warn_time = now
            return

        req = SdfQuery.Request()
        req.query_points = self.query_pts
        self.pending = True
        fut = self.client.call_async(req)
        fut.add_done_callback(self._on_response)


    def _on_response(self, future):
        self.pending = False
        try:
            res: SdfQuery.Response = future.result()
        except Exception as e:
            self.get_logger().error(f"SDF query call failed: {e}")
            return

        if not res.success:
            self.get_logger().warn("SDF query reported success=False")
            return

        n = len(res.signed_distances)
        if n != self.Nx * self.Ny:
            self.get_logger().warn(
                f"Response length mismatch: got {n}, expected {self.Nx*self.Ny}"
            )
            return

        # --- Extract fields
        sdf = np.asarray(res.signed_distances, dtype=np.float32)          # shape (n,)
        # Variances: either only SDF variance (length n) OR stacked rows (dim+1, n) when gradient variance is computed
        if len(res.variances) == n:
            var_sdf = np.asarray(res.variances, dtype=np.float32)
        else:
            # interpret as (rows, n), where first row is SDF variance
            rows = (len(res.variances) // n) if n > 0 else 1
            var_mat = np.asarray(res.variances, dtype=np.float64).reshape(rows, n)
            var_sdf = var_mat[0, :].astype(np.float32)

        # Build augmented scalar field phi = sdf - var_sdf (to match your current pipeline)
        phi = (sdf - var_sdf).reshape(self.Ny, self.Nx)  # row-major (y,x)

        # Gradients: prefer service-provided if available
        gx_grid = None
        gy_grid = None
        if res.compute_gradient and len(res.gradients) == n:
            # gradients are Vector3[] of length n; reshape to (Ny, Nx)
            gx = np.array([g.x for g in res.gradients], dtype=np.float32).reshape(self.Ny, self.Nx)
            gy = np.array([g.y for g in res.gradients], dtype=np.float32).reshape(self.Ny, self.Nx)
            gx_grid, gy_grid = gx, gy
        elif self.use_fallback_grad:
            # Fallback: finite-diff of phi on regular grid
            dy = float(self.ys[1] - self.ys[0]) if self.Ny > 1 else 1.0
            dx = float(self.xs[1] - self.xs[0]) if self.Nx > 1 else 1.0
            dVy, dVx = np.gradient(phi, dy, dx, edge_order=2)
            gx_grid = dVx.astype(np.float32)
            gy_grid = dVy.astype(np.float32)

        # --- Publish / Save exactly like your pointcloud node
        if self.mode == "pubsub":
            # Flatten as (grid.T).ravel() to match your published orientation
            vf_msg = ValueFunctionMsg()
            vf_msg.vf = (phi.T).ravel().tolist()
            self.vf_pub.publish(vf_msg)

            if gx_grid is not None and gy_grid is not None:
                gx_msg = ValueFunctionMsg(); gx_msg.vf = (gx_grid.T).ravel().tolist()
                gy_msg = ValueFunctionMsg(); gy_msg.vf = (gy_grid.T).ravel().tolist()
                self.gx_pub.publish(gx_msg)
                self.gy_pub.publish(gy_msg)

            if not self.first_info_sent:
                self._publish_info_once()
                # Save one snapshot (to mirror your original behavior)
                np.save(self.sdf_path, phi.T)
                if gx_grid is not None: np.save(self.gx_path, gx_grid.T)
                if gy_grid is not None: np.save(self.gy_path, gy_grid.T)
                self.first_info_sent = True

            # Publish info once every second   
            self.get_logger().info(
                f"Published VF {self.Nx}x{self.Ny}"
                + (", grads published (service)" if res.compute_gradient else
                   (", grads published (fallback)" if gx_grid is not None else ", no grads")),
            )

        else:  # file mode
            np.save(self.sdf_path, phi.T)
            if gx_grid is not None: np.save(self.gx_path, gx_grid.T)
            if gy_grid is not None: np.save(self.gy_path, gy_grid.T)
            b = Bool(); b.data = True
            self.vf_pub.publish(b); self.gx_pub.publish(b); self.gy_pub.publish(b)
            # Publish info once every second
            self.get_logger().info(
                f"Saved VF {self.Nx}x{self.Ny}"
                + (", grads saved (service)" if res.compute_gradient else
                   (", grads saved (fallback)" if gx_grid is not None else ", no grads")),
            )

    def _publish_info_once(self):
        if not self.info_pub:
            return
        meta = {
            "frame_id": "map",  # unknown here; adjust if you have a known frame source
            "xmin": float(self.xs[0]), "xmax": float(self.xs[-1]),
            "ymin": float(self.ys[0]), "ymax": float(self.ys[-1]),
            "nx": int(self.Nx), "ny": int(self.Ny),
            "dx": float(self.xs[1] - self.xs[0]) if self.Nx > 1 else float("nan"),
            "dy": float(self.ys[1] - self.ys[0]) if self.Ny > 1 else float("nan"),
            "source": f"service://{self.service_name}",
            "note": "Row-major (y,x). VF is SDF - var_sdf; gradients are of VF (service or fallback).",
        }
        self.info_pub.publish(String(data=json.dumps(meta)))

def main():
    rclpy.init()
    node = SDFServiceToGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

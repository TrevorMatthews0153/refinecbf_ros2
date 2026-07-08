#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import time
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Float64
from example_interfaces.msg import Bool
from refinecbf_ros2.msg import ValueFunctionMsg
from geometry_msgs.msg import Vector3
from erl_gp_sdf_msgs.srv import SdfQuery

try:
    from scipy import ndimage as _ndimg
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

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
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "state_domain" not in cfg:
        raise ValueError("YAML missing 'state_domain'")
    dom = cfg["state_domain"]
    lo  = np.array(dom["lo"],         dtype=np.float32)
    hi  = np.array(dom["hi"],         dtype=np.float32)
    res = np.array(dom["resolution"], dtype=np.int32)
    if not (len(lo) == len(hi) == len(res)):
        raise ValueError("lo, hi, resolution lengths must match")
    axes = [np.linspace(l, h, int(n), dtype=np.float32) for l, h, n in zip(lo, hi, res)]
    return {"lo": lo, "hi": hi, "resolution": res, "axes": axes}


def build_query_points_fast(ys_desc: np.ndarray, xs_desc: np.ndarray, z: float):
    """
    Vectorized replacement for the original Python-loop version.
    Builds (Ny*Nx,) Vector3 list: y outer (descending), x inner (descending).
    ~10-50x faster than the loop for large grids.
    """
    # meshgrid: Y varies in outer loop, X in inner -> indexing='ij' with y first
    yy, xx = np.meshgrid(ys_desc, xs_desc, indexing='ij')  # (Ny, Nx)
    yy_flat = yy.ravel().astype(np.float64)
    xx_flat = xx.ravel().astype(np.float64)
    n = len(yy_flat)
    pts = [None] * n
    for k in range(n):
        v = Vector3()
        v.x = xx_flat[k]
        v.y = yy_flat[k]
        v.z = z
        pts[k] = v
    return pts


def fill_missing(grid: np.ndarray) -> np.ndarray:
    """
    Fill NaNs by nearest valid value.
    If SciPy is available, use EDT-based nearest neighbor; else do a few
    4-neighbor averaging passes.
    """
    g = grid.copy()
    valid = np.isfinite(g)
    if np.all(valid):
        return g
    if _HAVE_SCIPY:
        nearest_idx = _ndimg.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        return g[tuple(nearest_idx)].astype(np.float32)
    # Fallback: iterative 4-neighbor averaging
    for _ in range(5):
        v = np.isfinite(g)
        if np.all(v):
            break
        g_pad = np.pad(g, 1, mode='edge')
        v_pad = np.pad(v, 1, mode='constant', constant_values=False)
        neigh = [
            (g_pad[:-2, 1:-1], v_pad[:-2, 1:-1]),
            (g_pad[ 2:, 1:-1], v_pad[ 2:, 1:-1]),
            (g_pad[1:-1, :-2], v_pad[1:-1, :-2]),
            (g_pad[1:-1, 2:],  v_pad[1:-1,  2:]),
        ]
        num = np.zeros_like(g, dtype=np.float32)
        den = np.zeros_like(g, dtype=np.int32)
        for c, m in neigh:
            mc = m & np.isfinite(c)
            num[mc] += c[mc].astype(np.float32)
            den[mc] += 1
        fill = (~v) & (den > 0)
        g[fill] = (num[fill] / den[fill]).astype(np.float32)
    return g.astype(np.float32)


# -------------------------
# Node
# -------------------------

class SDFServiceToGridNode(Node):
    """
    Queries erl_gp_sdf_msgs/SdfQuery and publishes/saves:
      - VF: phi = sdf - 2*sqrt(var_sdf)  (NaN where var_sdf >= threshold)

    Gradient computation is commented out but fully preserved for future use.
    To re-enable gradients, search for "# [GRAD]" and uncomment those lines.
    """

    def __init__(self):
        super().__init__("sdf_service_to_grid")

        # ---- Parameters (declare)
        self.declare_parameter("env_config_path",            "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/env.yaml")
        self.declare_parameter("service_name",               "sdf_query")
        self.declare_parameter("z",                          0.0)
        self.declare_parameter("publish_rate_hz",            10.0)
        self.declare_parameter("mode",                       "file")  # 'pubsub' or 'file'
        self.declare_parameter("output_topic",               "/env/sdf_update")
        # [GRAD] self.declare_parameter("output_topic_grad_x",  "/env/sdf_grad_x_update")
        # [GRAD] self.declare_parameter("output_topic_grad_y",  "/env/sdf_grad_y_update")
        self.declare_parameter("grid_info_topic",            "topics/sdf_grid_info")
        self.declare_parameter("sdf_file_path",              "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/sdf_sim.npy")
        # [GRAD] self.declare_parameter("grad_x_file_path",    "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/curr_grad_x.npy")
        # [GRAD] self.declare_parameter("grad_y_file_path",    "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/curr_grad_y.npy")
        # [GRAD] self.declare_parameter("use_fallback_gradient_when_missing", False)
        self.declare_parameter("default_invalid_sdf",        -0.005)
        self.declare_parameter("invalid_variance_threshold", 1.0e5)
        self.declare_parameter("match_viz_query_order",      True)
        self.declare_parameter("fill_missing",               True)
        self.declare_parameter("save_sdf",                   False)

        # ---- Parameters (read)
        env_path                  = self.get_parameter("env_config_path").value
        self.service_name         = self.get_parameter("service_name").value
        self.z                    = float(self.get_parameter("z").value)
        hz                        = float(self.get_parameter("publish_rate_hz").value)
        self.mode                 = self.get_parameter("mode").value.lower()
        out_vf_topic              = self.get_parameter("output_topic").value
        # [GRAD] out_gx_topic     = self.get_parameter("output_topic_grad_x").value
        # [GRAD] out_gy_topic     = self.get_parameter("output_topic_grad_y").value
        info_topic                = self.get_parameter("grid_info_topic").value
        self.sdf_path             = self.get_parameter("sdf_file_path").value
        # [GRAD] self.gx_path     = self.get_parameter("grad_x_file_path").value
        # [GRAD] self.gy_path     = self.get_parameter("grad_y_file_path").value
        # [GRAD] self.use_fallback_grad = bool(self.get_parameter("use_fallback_gradient_when_missing").value)
        self.default_invalid_sdf  = float(self.get_parameter("default_invalid_sdf").value)
        self.invalid_var_thresh   = float(self.get_parameter("invalid_variance_threshold").value)
        match_viz_query_order     = bool(self.get_parameter("match_viz_query_order").value)
        self.fill_missing_flag    = bool(self.get_parameter("fill_missing").value)
        self.save_sdf             = bool(self.get_parameter("save_sdf").value)

        self.current_goals_reached = 0
        self.save_dir = "/root/ros2_ws/noise_and_range_experiments/low_range_high_noise"
        if self.save_sdf:
            self.goal_reached_sub = self.create_subscription(
                Bool, "goal_reached", self.goal_reached_cb, 1
            )

        # ---- Load env grid (x,y only)
        dom      = parse_env_yaml(env_path)
        xs_asc   = dom["axes"][0]
        ys_asc   = dom["axes"][1]
        self.Nx  = int(xs_asc.size)
        self.Ny  = int(ys_asc.size)
        self.N   = self.Nx * self.Ny

        if match_viz_query_order:
            self.xs_for_query = xs_asc[::-1]
            self.ys_for_query = ys_asc[::-1]
        else:
            self.xs_for_query = xs_asc
            self.ys_for_query = ys_asc
        self.xs = xs_asc
        self.ys = ys_asc

        # Pre-allocate reusable work arrays (avoids repeated malloc in the hot path)
        self._sdf_buf   = np.empty(self.N, dtype=np.float32)
        self._var_buf   = np.empty(self.N, dtype=np.float32)
        self._phi_buf   = np.empty(self.N, dtype=np.float32)
        self._valid_buf = np.empty(self.N, dtype=bool)

        # ---- Publishers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # [GRAD] Pass out_gx_topic, out_gy_topic into _create_publishers when re-enabling
        self._create_publishers(self.mode, out_vf_topic, info_topic, qos)
        self.timing_pub  = self.create_publisher(Float64, "gp_query_node_time_ms", qos)
        self.first_info_sent = False

        # ---- Service client
        self.client = self.create_client(SdfQuery, self.service_name)

        # Prebuild query points (vectorized)
        self.query_pts = build_query_points_fast(
            self.ys_for_query, self.xs_for_query, self.z
        )

        # ---- Timer
        self.period           = 1.0 / max(1e-6, hz)
        self.pending          = False
        self._query_send_time = None
        self.timer            = self.create_timer(self.period, self._tick)
        self._last_warn_time  = None

        self.get_logger().info(
            f"Ready. Mode={self.mode} service='{self.service_name}' grid=({self.Ny}x{self.Nx}) "
            f"x in [{self.xs[0]:.2f},{self.xs[-1]:.2f}] y in [{self.ys[0]:.2f},{self.ys[-1]:.2f}] z={self.z:.2f} "
            f"(invalid_var_thresh={self.invalid_var_thresh:g}, match_viz_query_order={match_viz_query_order})"
        )

    def _create_publishers(self, mode, out_vf_topic, info_topic, qos):
        # [GRAD] Also accept out_gx_topic, out_gy_topic as arguments when re-enabling
        if mode == "pubsub":
            self.vf_pub   = self.create_publisher(ValueFunctionMsg, out_vf_topic, qos)
            # [GRAD] self.gx_pub = self.create_publisher(ValueFunctionMsg, out_gx_topic, qos)
            # [GRAD] self.gy_pub = self.create_publisher(ValueFunctionMsg, out_gy_topic, qos)
            self.info_pub = self.create_publisher(String, info_topic, qos)
        elif mode == "file":
            self.vf_pub   = self.create_publisher(Bool, out_vf_topic, qos)
            # [GRAD] self.gx_pub = self.create_publisher(Bool, out_gx_topic, qos)
            # [GRAD] self.gy_pub = self.create_publisher(Bool, out_gy_topic, qos)
            self.info_pub = None
        else:
            self.get_logger().warn(f"Unknown mode '{mode}', defaulting to 'pubsub'")
            self.mode     = "pubsub"
            self.vf_pub   = self.create_publisher(ValueFunctionMsg, out_vf_topic, qos)
            # [GRAD] self.gx_pub = self.create_publisher(ValueFunctionMsg, out_gx_topic, qos)
            # [GRAD] self.gy_pub = self.create_publisher(ValueFunctionMsg, out_gy_topic, qos)
            self.info_pub = self.create_publisher(String, info_topic, qos)

    # --------- Internals ----------

    def _tick(self):
        if self.pending:
            return
        if not self.client.service_is_ready():
            now = self.get_clock().now()
            if (self._last_warn_time is None) or (
                (now - self._last_warn_time).nanoseconds * 1e-9 > 5.0
            ):
                self.get_logger().warn(f"Service '{self.service_name}' not ready")
                self._last_warn_time = now
            return
        req = SdfQuery.Request()
        req.query_points = self.query_pts
        self.pending = True
        self._query_send_time = time.perf_counter()
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
        if n != self.N:
            self.get_logger().warn(f"Response length mismatch: got {n}, expected {self.N}")
            return

        # --- Extract into pre-allocated buffers (avoids new allocs)
        sdf = self._sdf_buf
        np.copyto(sdf, res.signed_distances, casting='unsafe')

        # Variances: either length n (SDF only) or stacked rows ((dim+1)*n)
        nv = len(res.variances)
        if nv == n:
            np.copyto(self._var_buf, res.variances, casting='unsafe')
        else:
            # Column-major stacked: first row is SDF variance
            rows = nv // n if n > 0 else 1
            var_mat = np.frombuffer(bytes(res.variances), dtype=np.float64) \
                        .reshape(rows, n, order="F")
            self._var_buf[:] = var_mat[0, :].astype(np.float32)
        var_sdf = self._var_buf

        # --- Validity mask (in-place, no new array)
        valid = self._valid_buf
        np.less(var_sdf, self.invalid_var_thresh, out=valid)

        # Pre-compute invalid_sdf sentinel mask
        invalid_sdf_mask = (sdf == self.default_invalid_sdf)  # (n,) bool

        # --- phi = sdf - 2*sqrt(var_sdf), NaN where invalid (all in-place)
        phi = self._phi_buf
        np.sqrt(var_sdf, out=phi)       # phi = sqrt(var)
        phi *= 2.0                      # phi = 2*sqrt(var)
        np.subtract(sdf, phi, out=phi)  # phi = sdf - 2*sqrt(var)
        phi[~valid] = np.nan            # mask invalid cells

        # --- Reshape (Ny, Nx) and un-flip the descending query order
        # Queried with y descending, x descending -> flipud+fliplr restores ascending
        phi_grid = phi.reshape(self.Ny, self.Nx)
        phi_grid = np.flipud(np.fliplr(phi_grid))
        phi_grid = phi_grid.astype(np.float32, copy=False)

        # --- Fill NaNs if requested
        if self.fill_missing_flag and not np.all(np.isfinite(phi_grid)):
            phi_grid = fill_missing(phi_grid)

        # Re-stamp sentinel on invalid cells
        phi_grid[invalid_sdf_mask.reshape(self.Ny, self.Nx)] = self.default_invalid_sdf

        # --- [GRAD] Gradient computation (service-provided or fallback finite-diff)
        # gx_grid = None
        # gy_grid = None
        # if getattr(res, "compute_gradient", False) and len(res.gradients) == n:
        #     gx = np.array([g.x for g in res.gradients], dtype=np.float32)
        #     gy = np.array([g.y for g in res.gradients], dtype=np.float32)
        #     gx = np.where(valid, gx, np.nan).astype(np.float32)
        #     gy = np.where(valid, gy, np.nan).astype(np.float32)
        #     gx_grid = np.flipud(np.fliplr(gx.reshape(self.Ny, self.Nx)))
        #     gy_grid = np.flipud(np.fliplr(gy.reshape(self.Ny, self.Nx)))
        # elif self.use_fallback_grad:
        #     dy = float(abs(self.ys[1] - self.ys[0])) if self.Ny > 1 else 1.0
        #     dx = float(abs(self.xs[1] - self.xs[0])) if self.Nx > 1 else 1.0
        #     dVy, dVx = np.gradient(phi_grid, dy, dx, edge_order=2)
        #     gx_grid = dVx.astype(np.float32)
        #     gy_grid = dVy.astype(np.float32)

        # --- Timing
        elapsed_ms = (time.perf_counter() - self._query_send_time) * 1000.0
        timing_msg = Float64()
        timing_msg.data = elapsed_ms
        self.timing_pub.publish(timing_msg)
        self.get_logger().info(f"SDF compute time: {elapsed_ms:.1f} ms")

        # phi_out is (Nx, Ny) in column-major publish convention
        phi_out = phi_grid.T  # zero-copy view

        # --- Publish / Save
        if self.mode == "pubsub":
            vf_msg = ValueFunctionMsg()
            vf_msg.vf = phi_out.ravel(order="C").tolist()
            self.vf_pub.publish(vf_msg)

            # [GRAD] if gx_grid is not None and gy_grid is not None:
            # [GRAD]     gx_msg = ValueFunctionMsg(); gx_msg.vf = (gx_grid.T).ravel().tolist()
            # [GRAD]     gy_msg = ValueFunctionMsg(); gy_msg.vf = (gy_grid.T).ravel().tolist()
            # [GRAD]     self.gx_pub.publish(gx_msg)
            # [GRAD]     self.gy_pub.publish(gy_msg)

            if not self.first_info_sent:
                self._publish_info_once()
                self.first_info_sent = True

            np.save(self.sdf_path, phi_out)
            # [GRAD] if gx_grid is not None: np.save(self.gx_path, gx_grid.T)
            # [GRAD] if gy_grid is not None: np.save(self.gy_path, gy_grid.T)

            self.get_logger().info(f"Published VF {self.Nx}x{self.Ny}")

        else:  # file mode
            np.save(self.sdf_path, phi_out)
            # [GRAD] if gx_grid is not None: np.save(self.gx_path, gx_grid.T)
            # [GRAD] if gy_grid is not None: np.save(self.gy_path, gy_grid.T)

            b = Bool(); b.data = True
            self.vf_pub.publish(b)
            # [GRAD] self.gx_pub.publish(b)
            # [GRAD] self.gy_pub.publish(b)

            self.get_logger().info(f"Saved VF {self.Nx}x{self.Ny}")

    def _publish_info_once(self):
        if not self.info_pub:
            return
        meta = {
            "frame_id": "map",
            "xmin": float(self.xs[0]),  "xmax": float(self.xs[-1]),
            "ymin": float(self.ys[0]),  "ymax": float(self.ys[-1]),
            "nx": int(self.Nx),         "ny": int(self.Ny),
            "dx": float(self.xs[1] - self.xs[0]) if self.Nx > 1 else float("nan"),
            "dy": float(self.ys[1] - self.ys[0]) if self.Ny > 1 else float("nan"),
            "source": f"service://{self.service_name}",
            "note": "VF = SDF - 2*sqrt(var_sdf); NaNs where var_sdf >= threshold.",
        }
        self.info_pub.publish(String(data=json.dumps(meta)))

    def goal_reached_cb(self, msg):
        if msg.data and self.save_sdf:
            self.get_logger().info("Goal reached, saving SDF")
            self.current_goals_reached += 1
            np.save(
                f"{self.save_dir}/goal_{self.current_goals_reached}.npy",
                np.load(self.sdf_path)
            )
            self.get_logger().info(
                f"Saved SDF to {self.save_dir}/goal_{self.current_goals_reached}_sdf.npy"
            )


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


#=========STALE=========
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import json
# import time
# import yaml
# import numpy as np

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# from std_msgs.msg import String, Float64
# from example_interfaces.msg import Bool
# from refinecbf_ros2.msg import ValueFunctionMsg

# from geometry_msgs.msg import Vector3
# from erl_gp_sdf_msgs.srv import SdfQuery

# try:
#     from scipy import ndimage as _ndimg
#     _HAVE_SCIPY = True
# except Exception:
#     _HAVE_SCIPY = False


# # -------------------------
# # Helpers
# # -------------------------

# def parse_env_yaml(path):
#     """
#     Expect:
#       state_domain:
#         lo: [x_min, y_min, ...]
#         hi: [x_max, y_max, ...]
#         resolution: [nx, ny, ...]
#     """
#     with open(path, "r") as f:
#         cfg = yaml.safe_load(f)
#     if "state_domain" not in cfg:
#         raise ValueError("YAML missing 'state_domain'")

#     dom = cfg["state_domain"]
#     lo = np.array(dom["lo"], dtype=np.float32)
#     hi = np.array(dom["hi"], dtype=np.float32)
#     res = np.array(dom["resolution"], dtype=np.int32)
#     if not (len(lo) == len(hi) == len(res)):
#         raise ValueError("lo, hi, resolution lengths must match")

#     axes = [np.linspace(l, h, int(n), dtype=np.float32) for l, h, n in zip(lo, hi, res)]
#     return {"lo": lo, "hi": hi, "resolution": res, "axes": axes}


# def flatten_grid_column_major_like_viz(ys_desc: np.ndarray, xs_desc: np.ndarray, z: float):
#     """
#     Build query points to mirror the C++ viz node:
#       for (j = +half_y .. -half_y)     // y outer (descending)
#         for (i = +half_x .. -half_x)   // x inner (descending), x is fastest
#     """
#     pts = []
#     for y in ys_desc:
#         for x in xs_desc:
#             v = Vector3()
#             v.x = float(x)
#             v.y = float(y)
#             v.z = float(z)
#             pts.append(v)
#     return pts


# def fill_missing(grid: np.ndarray) -> np.ndarray:
#     """
#     Fill NaNs by nearest valid value (same idea as the point-cloud subscriber).
#     If SciPy is available, use EDT-based nearest neighbor; else do a few
#     4-neighbor averaging passes.
#     """
#     g = grid.copy()
#     valid = np.isfinite(g)
#     if np.all(valid):
#         return g

#     if _HAVE_SCIPY:
#         nearest_idx = _ndimg.distance_transform_edt(
#             ~valid, return_distances=False, return_indices=True
#         )
#         return g[tuple(nearest_idx)].astype(np.float32)

#     # Fallback: iterative 4-neighbor averaging
#     for _ in range(5):
#         v = np.isfinite(g)
#         if np.all(v):
#             break
#         g_pad = np.pad(g, 1, mode='edge')
#         v_pad = np.pad(v, 1, mode='constant', constant_values=False)
#         neigh = [
#             (g_pad[:-2, 1:-1], v_pad[:-2, 1:-1]),
#             (g_pad[ 2:, 1:-1], v_pad[ 2:, 1:-1]),
#             (g_pad[1:-1, :-2], v_pad[1:-1, :-2]),
#             (g_pad[1:-1,  2:], v_pad[1:-1,  2:]),
#         ]
#         num = np.zeros_like(g, dtype=np.float32)
#         den = np.zeros_like(g, dtype=np.int32)
#         for c, m in neigh:
#             mc = m & np.isfinite(c)
#             num[mc] += c[mc].astype(np.float32)
#             den[mc] += 1
#         fill = (~v) & (den > 0)
#         g[fill] = (num[fill] / den[fill]).astype(np.float32)
#     return g.astype(np.float32)


# # -------------------------
# # Node
# # -------------------------

# class SDFServiceToGridNode(Node):
#     """
#     Queries erl_gp_sdf_msgs/SdfQuery and publishes/saves:
#       - VF: phi = sdf - var_sdf  (NaN where var_sdf >= threshold)
#       - Gradients (service if present, else optional fallback)
#     """

#     def __init__(self):
#         super().__init__("sdf_service_to_grid")

#         # ---- Parameters (declare)
#         self.declare_parameter("env_config_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/env.yaml")
#         self.declare_parameter("service_name", "sdf_query")
#         self.declare_parameter("z", 0.0)
#         self.declare_parameter("publish_rate_hz", 10.0)
#         self.declare_parameter("mode", "file")  # 'pubsub' or 'file'
#         self.declare_parameter("output_topic", "/env/sdf_update")
#         self.declare_parameter("output_topic_grad_x", "/env/sdf_grad_x_update")
#         self.declare_parameter("output_topic_grad_y", "/env/sdf_grad_y_update")
#         self.declare_parameter("grid_info_topic", "topics/sdf_grid_info")
#         self.declare_parameter("sdf_file_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/sdf_sim.npy")
#         self.declare_parameter("grad_x_file_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/curr_grad_x.npy")
#         self.declare_parameter("grad_y_file_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/curr_grad_y.npy")
#         self.declare_parameter("use_fallback_gradient_when_missing", False)
#         self.declare_parameter("default_invalid_sdf", -0.005)  # must match GP setting
#         self.declare_parameter("invalid_variance_threshold", 1.0e5)
#         self.declare_parameter("match_viz_query_order", True)
#         self.declare_parameter("fill_missing", True)
#         self.declare_parameter("save_sdf", False)

#         # ---- Parameters (read)
#         env_path = self.get_parameter("env_config_path").value
#         self.service_name = self.get_parameter("service_name").value
#         self.z = float(self.get_parameter("z").value)
#         hz = float(self.get_parameter("publish_rate_hz").value)
#         self.mode = self.get_parameter("mode").value.lower()

#         out_vf_topic = self.get_parameter("output_topic").value
#         out_gx_topic = self.get_parameter("output_topic_grad_x").value
#         out_gy_topic = self.get_parameter("output_topic_grad_y").value
#         info_topic   = self.get_parameter("grid_info_topic").value

#         self.sdf_path = self.get_parameter("sdf_file_path").value
#         self.gx_path  = self.get_parameter("grad_x_file_path").value
#         self.gy_path  = self.get_parameter("grad_y_file_path").value

#         self.use_fallback_grad = bool(self.get_parameter("use_fallback_gradient_when_missing").value)
#         self.default_invalid_sdf = float(self.get_parameter("default_invalid_sdf").value)
#         self.invalid_var_thresh = float(self.get_parameter("invalid_variance_threshold").value)
#         match_viz_query_order = bool(self.get_parameter("match_viz_query_order").value)
#         self.fill_missing = bool(self.get_parameter("fill_missing").value)

#         self.save_sdf = bool(self.get_parameter("save_sdf").value)
#         self.current_goals_reached = 0
#         self.save_dir = "/root/ros2_ws/noise_and_range_experiments/low_range_high_noise"

#         if self.save_sdf:
#             self.goal_reached_sub = self.create_subscription(Bool, "goal_reached", self.goal_reached_cb, 1)

#         # ---- Load env grid (x,y only)
#         dom = parse_env_yaml(env_path)
#         xs_asc = dom["axes"][0]
#         ys_asc = dom["axes"][1]
#         self.Nx = int(xs_asc.size)
#         self.Ny = int(ys_asc.size)

#         if match_viz_query_order:
#             self.xs_for_query = xs_asc[::-1]
#             self.ys_for_query = ys_asc[::-1]
#         else:
#             self.xs_for_query = xs_asc
#             self.ys_for_query = ys_asc

#         self.xs = xs_asc
#         self.ys = ys_asc

#         # ---- Publishers
#         qos = QoSProfile(
#             reliability=ReliabilityPolicy.RELIABLE,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=5,
#         )
#         self._create_publishers(self.mode, out_vf_topic, out_gx_topic, out_gy_topic, info_topic, qos)

#         self.timing_pub = self.create_publisher(Float64, "gp_query_node_time_ms", qos)
#         self.first_info_sent = False

#         # ---- Service client
#         self.client = self.create_client(SdfQuery, self.service_name)

#         # Prebuild query points with viz-like order (y outer, x inner, both descending)
#         self.query_pts = flatten_grid_column_major_like_viz(self.ys_for_query, self.xs_for_query, self.z)

#         # ---- Timer
#         self.period = 1.0 / max(1e-6, hz)
#         self.pending = False
#         self._query_send_time = None   # wall-clock time when query was sent
#         self.timer = self.create_timer(self.period, self._tick)
#         self._last_warn_time = None

#         self.get_logger().info(
#             f"Ready. Mode={self.mode} service='{self.service_name}' grid=({self.Ny}x{self.Nx}) "
#             f"x∈[{self.xs[0]:.2f},{self.xs[-1]:.2f}] y∈[{self.ys[0]:.2f},{self.ys[-1]:.2f}] z={self.z:.2f} "
#             f"(invalid_var_thresh={self.invalid_var_thresh:g}, match_viz_query_order={match_viz_query_order})"
#         )

#     def _create_publishers(self, mode, out_vf_topic, out_gx_topic, out_gy_topic, info_topic, qos):
#         """
#         Create publishers based on mode.
#         This is a cleanup-only helper to remove repetition.
#         """
#         if mode == "pubsub":
#             self.vf_pub = self.create_publisher(ValueFunctionMsg, out_vf_topic, qos)
#             self.gx_pub = self.create_publisher(ValueFunctionMsg, out_gx_topic, qos)
#             self.gy_pub = self.create_publisher(ValueFunctionMsg, out_gy_topic, qos)
#             self.info_pub = self.create_publisher(String, info_topic, qos)
#         elif mode == "file":
#             self.vf_pub = self.create_publisher(Bool, out_vf_topic, qos)
#             self.gx_pub = self.create_publisher(Bool, out_gx_topic, qos)
#             self.gy_pub = self.create_publisher(Bool, out_gy_topic, qos)
#             self.info_pub = None
#         else:
#             self.get_logger().warn(f"Unknown mode '{mode}', defaulting to 'pubsub'")
#             self.mode = "pubsub"
#             self.vf_pub = self.create_publisher(ValueFunctionMsg, out_vf_topic, qos)
#             self.gx_pub = self.create_publisher(ValueFunctionMsg, out_gx_topic, qos)
#             self.gy_pub = self.create_publisher(ValueFunctionMsg, out_gy_topic, qos)
#             self.info_pub = self.create_publisher(String, info_topic, qos)

#     # --------- Internals ----------

#     def _tick(self):
#         if self.pending:
#             return

#         if not self.client.service_is_ready():
#             now = self.get_clock().now()
#             if (self._last_warn_time is None) or ((now - self._last_warn_time).nanoseconds * 1e-9 > 5.0):
#                 self.get_logger().warn(f"Service '{self.service_name}' not ready")
#                 self._last_warn_time = now
#             return

#         req = SdfQuery.Request()
#         req.query_points = self.query_pts
#         self.pending = True
#         self._query_send_time = time.perf_counter()   # record query send time
#         fut = self.client.call_async(req)
#         fut.add_done_callback(self._on_response)

#     def _on_response(self, future):
#         self.pending = False
#         try:
#             res: SdfQuery.Response = future.result()
#         except Exception as e:
#             self.get_logger().error(f"SDF query call failed: {e}")
#             return

#         if not res.success:
#             self.get_logger().warn("SDF query reported success=False")
#             return

#         n = len(res.signed_distances)
#         if n != self.Nx * self.Ny:
#             self.get_logger().warn(f"Response length mismatch: got {n}, expected {self.Nx*self.Ny}")
#             return

#         # --- Extract arrays
#         sdf = np.asarray(res.signed_distances, dtype=np.float32)  # (n,)

#         # Variances layout: either length n (only SDF var) or stacked rows ((dim+1)*n)
#         if len(res.variances) == n:
#             var_sdf = np.asarray(res.variances, dtype=np.float32)
#         else:
#             rows = (len(res.variances) // n) if n > 0 else 1
#             var_mat = np.asarray(res.variances, dtype=np.float64).reshape(rows, n, order="F")
#             var_sdf = var_mat[0, :].astype(np.float32)

#         # ----- Mask invalid (mirror viz node: treat huge variance as invalid)
#         valid = var_sdf < self.invalid_var_thresh
#         invalid_sdf_mask = sdf == self.default_invalid_sdf
#         # Keep NaN where invalid so downstream can ignore
#         sdf_masked = np.where(valid, sdf, np.nan).astype(np.float32)
#         var_sdf_masked = np.where(valid, var_sdf, np.nan).astype(np.float32)

#         # Build augmented scalar field phi = sdf - var_sdf (NaN where invalid)
        
#         ### For SDF-GP
#         phi_flat = sdf_masked - 2 * np.sqrt(var_sdf_masked)  # (n,)
        
#         # Reshape to (Ny, Nx) *in the order we queried*
#         # We queried with y descending and x descending, x fastest; we keep (Ny,Nx) for internal work,
#         # then publish as (phi.T).ravel() to match your original topic orientation.
#         phi_grid = phi_flat.reshape(self.Ny, self.Nx)
#         phi_grid = np.flipud(np.fliplr(phi_grid))

#         valid_mask_before_fill = np.isfinite(phi_grid)
#         if self.fill_missing and not np.all(valid_mask_before_fill):
#             phi_grid_filled = fill_missing(phi_grid)
#         else:
#             phi_grid_filled = phi_grid

#         phi_grid_filled[invalid_sdf_mask.reshape(self.Ny, self.Nx)] = self.default_invalid_sdf

#         # self.get_logger().info(
#         #     f"SDF range: {float(np.nanmin(sdf_masked)):.4f} to {float(np.nanmax(sdf_masked)):.4f} m")
#         # self.get_logger().info(
#         #     f"var range: {float(np.nanmin(var_sdf_masked)):.6f} to {float(np.nanmax(var_sdf_masked)):.6f}")
#         # self.get_logger().info(
#         #     f"sqrt(var) range: {float(np.nanmin(np.sqrt(var_sdf_masked))):.6f} to {float(np.nanmax(np.sqrt(var_sdf_masked))):.6f}")
#         # self.get_logger().info(
#         #     f"2*sqrt(var) range: {float(np.nanmin(2*np.sqrt(var_sdf_masked))):.6f} to {float(np.nanmax(2*np.sqrt(var_sdf_masked))):.6f}")
        
#         # boundary_mask = (np.abs(sdf_masked) < 0.5) & np.isfinite(var_sdf_masked)
#         # if boundary_mask.any():
#         #     boundary_var = var_sdf_masked[boundary_mask]
#         #     self.get_logger().info(
#         #         f"Boundary (|SDF|<0.5m) 2*sqrt(var): "
#         #         f"min={float(2*np.sqrt(boundary_var.min())):.3f} "
#         #         f"mean={float(2*np.sqrt(boundary_var.mean())):.3f} "
#         #         f"max={float(2*np.sqrt(boundary_var.max())):.3f} m")

#         # Gradients: prefer service-provided if available
#         gx_grid = None
#         gy_grid = None
#         if getattr(res, "compute_gradient", False) and len(res.gradients) == n:
#             gx = np.array([g.x for g in res.gradients], dtype=np.float32)
#             gy = np.array([g.y for g in res.gradients], dtype=np.float32)
#             # Mask invalid grads as well
#             gx = np.where(valid, gx, np.nan).astype(np.float32)
#             gy = np.where(valid, gy, np.nan).astype(np.float32)
#             gx_grid = gx.reshape(self.Ny, self.Nx)
#             gy_grid = gy.reshape(self.Ny, self.Nx)
#         elif self.use_fallback_grad:
#             # Fallback finite-diff on regular grid (use ascending spacings)
#             dy = float(abs(self.ys[1] - self.ys[0])) if self.Ny > 1 else 1.0
#             dx = float(abs(self.xs[1] - self.xs[0])) if self.Nx > 1 else 1.0
#             dVy, dVx = np.gradient(phi_grid_filled, dy, dx, edge_order=2)
#             gx_grid = dVx.astype(np.float32)
#             gy_grid = dVy.astype(np.float32)

#         # --- Publish timing: query send → output ready
#         elapsed_ms = (time.perf_counter() - self._query_send_time) * 1000.0
#         timing_msg = Float64()
#         timing_msg.data = elapsed_ms
#         self.timing_pub.publish(timing_msg)
#         self.get_logger().info(f"SDF compute time: {elapsed_ms:.1f} ms")

#         # --- Publish / Save
#         if self.mode == "pubsub":
#             vf_msg = ValueFunctionMsg()
#             vf_msg.vf = (phi_grid_filled.T).ravel(order="C").tolist()
#             self.vf_pub.publish(vf_msg)

#             if gx_grid is not None and gy_grid is not None:
#                 gx_msg = ValueFunctionMsg(); gx_msg.vf = (gx_grid.T).ravel().tolist()
#                 gy_msg = ValueFunctionMsg(); gy_msg.vf = (gy_grid.T).ravel().tolist()
#                 self.gx_pub.publish(gx_msg)
#                 self.gy_pub.publish(gy_msg)

#             if not self.first_info_sent:
#                 self._publish_info_once()
#                 # Snapshot saves
#                 np.save(self.sdf_path, phi_grid_filled.T)
#                 if gx_grid is not None: np.save(self.gx_path, gx_grid.T)
#                 if gy_grid is not None: np.save(self.gy_path, gy_grid.T)
#                 self.first_info_sent = True

#             self.get_logger().info(
#                 f"Published VF {self.Nx}x{self.Ny}"
#                 + (", grads (service)" if getattr(res, "compute_gradient", False)
#                    else (", grads (fallback)" if gx_grid is not None else ", no grads"))
#             )

#         else:  # file mode
#             np.save(self.sdf_path, phi_grid_filled.T)
#             if gx_grid is not None: np.save(self.gx_path, gx_grid.T)
#             if gy_grid is not None: np.save(self.gy_path, gy_grid.T)
#             b = Bool(); b.data = True
#             self.vf_pub.publish(b); self.gx_pub.publish(b); self.gy_pub.publish(b)
#             self.get_logger().info(
#                 f"Saved VF {self.Nx}x{self.Ny}"
#                 + (", grads (service)" if getattr(res, "compute_gradient", False)
#                    else (", grads (fallback)" if gx_grid is not None else ", no grads"))
#             )

#     def _publish_info_once(self):
#         if not self.info_pub:
#             return
#         meta = {
#             "frame_id": "map",
#             "xmin": float(self.xs[0]), "xmax": float(self.xs[-1]),
#             "ymin": float(self.ys[0]), "ymax": float(self.ys[-1]),
#             "nx": int(self.Nx), "ny": int(self.Ny),
#             "dx": float(self.xs[1] - self.xs[0]) if self.Nx > 1 else float("nan"),
#             "dy": float(self.ys[1] - self.ys[0]) if self.Ny > 1 else float("nan"),
#             "source": f"service://{self.service_name}",
#             "note": "Query order matches viz node (descending y,x). VF = SDF - var_sdf; NaNs where var_sdf >= threshold.",
#         }
#         self.info_pub.publish(String(data=json.dumps(meta)))

#     def goal_reached_cb(self, msg):
#         if msg.data and self.save_sdf:
#             self.get_logger().info("Goal reached, saving SDF")
#             self.current_goals_reached += 1
#             np.save(f"{self.save_dir}/goal_{self.current_goals_reached}.npy", np.load(self.sdf_path))
#             self.get_logger().info(f"Saved SDF to {self.save_dir}/goal_{self.current_goals_reached}_sdf.npy")


# def main():
#     rclpy.init()
#     node = SDFServiceToGridNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
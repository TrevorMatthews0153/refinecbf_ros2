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

from isdf_msgs.srv import ISdfQuery  # adjust if your generated name differs

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
    """
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
    Queries iSDF ISdfQuery and publishes/saves:
      - VF: phi = signed_distance (raw), optionally filling sentinel invalid cells.

    Speedup: query_stride>1 queries a strided (coarser) grid and upsamples back to
    the full grid with nearest-neighbor repeat.
    """

    def __init__(self):
        super().__init__("sdf_service_to_grid")

        # ---- Parameters (declare)
        self.declare_parameter("env_config_path",            "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/env.yaml")
        self.declare_parameter("service_name",               "/isdf_train/sdf_query")
        self.declare_parameter("z",                          0.060)
        self.declare_parameter("publish_rate_hz",            10.0)
        self.declare_parameter("mode",                       "file")  # 'pubsub' or 'file'
        self.declare_parameter("output_topic",               "/env/sdf_update")
        self.declare_parameter("grid_info_topic",            "topics/sdf_grid_info")
        self.declare_parameter("sdf_file_path",              "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/sdf_sim.npy")
        self.declare_parameter("default_invalid_sdf",        -0.005)
        self.declare_parameter("match_viz_query_order",      True)
        self.declare_parameter("fill_missing",               True)
        self.declare_parameter("save_sdf",                   False)

        # ---- Speedup parameter (NEW)
        self.declare_parameter("query_stride",               2)

        # ---- Parameters (read)
        env_path                  = self.get_parameter("env_config_path").value
        self.service_name         = self.get_parameter("service_name").value
        self.z                    = float(self.get_parameter("z").value)
        hz                        = float(self.get_parameter("publish_rate_hz").value)
        self.mode                 = self.get_parameter("mode").value.lower()
        out_vf_topic              = self.get_parameter("output_topic").value
        info_topic                = self.get_parameter("grid_info_topic").value
        self.sdf_path             = self.get_parameter("sdf_file_path").value
        self.default_invalid_sdf  = float(self.get_parameter("default_invalid_sdf").value)
        match_viz_query_order     = bool(self.get_parameter("match_viz_query_order").value)
        self.fill_missing_flag    = bool(self.get_parameter("fill_missing").value)
        self.save_sdf             = bool(self.get_parameter("save_sdf").value)

        # ---- Speedup (read)
        self.query_stride = int(self.get_parameter("query_stride").value)
        self.query_stride = max(1, self.query_stride)

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

        # ---- Strided query grid (NEW)
        xs_q = self.xs_for_query[::self.query_stride]
        ys_q = self.ys_for_query[::self.query_stride]
        self.Nx_q = int(xs_q.size)
        self.Ny_q = int(ys_q.size)
        self.N_q  = self.Nx_q * self.Ny_q

        # Prebuild query points (vectorized) at coarse resolution
        self.query_pts = build_query_points_fast(ys_q, xs_q, self.z)
        self.get_logger().info(
            f"stride={self.query_stride} Nx_q={self.Nx_q} Ny_q={self.Ny_q} "
            f"N_q={self.N_q} len(query_pts)={len(self.query_pts)}"
        )

        # Pre-allocate reusable work arrays sized for COARSE query
        self._sdf_buf   = np.empty(self.N_q, dtype=np.float32)
        self._phi_buf   = np.empty(self.N_q, dtype=np.float32)

        # ---- Publishers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._create_publishers(self.mode, out_vf_topic, info_topic, qos)
        self.timing_pub  = self.create_publisher(Float64, "gp_query_node_time_ms", qos)
        self.first_info_sent = False

        # ---- Service client
        self.client = self.create_client(ISdfQuery, self.service_name)

        # ---- Timer
        self.period           = 1.0 / max(1e-6, hz)
        self.pending          = False
        self._query_send_time = None
        self.timer            = self.create_timer(self.period, self._tick)
        self._last_warn_time  = None

        self.get_logger().info(
            f"Ready. Mode={self.mode} service='{self.service_name}' full_grid=({self.Ny}x{self.Nx})={self.N} "
            f"query_stride={self.query_stride} -> query_grid=({self.Ny_q}x{self.Nx_q})={self.N_q} z={self.z:.2f}"
        )

    def _create_publishers(self, mode, out_vf_topic, info_topic, qos):
        if mode == "pubsub":
            self.vf_pub   = self.create_publisher(ValueFunctionMsg, out_vf_topic, qos)
            self.info_pub = self.create_publisher(String, info_topic, qos)
        elif mode == "file":
            self.vf_pub   = self.create_publisher(Bool, out_vf_topic, qos)
            self.info_pub = None
        else:
            self.get_logger().warn(f"Unknown mode '{mode}', defaulting to 'pubsub'")
            self.mode     = "pubsub"
            self.vf_pub   = self.create_publisher(ValueFunctionMsg, out_vf_topic, qos)
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

        req = ISdfQuery.Request()
        req.query_points = self.query_pts

        self.pending = True
        self.get_logger().info(f"Calling iSDF with N_q={self.N_q} points (stride={self.query_stride})")
        self._query_send_time = time.perf_counter()
        fut = self.client.call_async(req)
        fut.add_done_callback(self._on_response)

    def _on_response(self, future):
        self.pending = False
        try:
            res: ISdfQuery.Response = future.result()
            self.get_logger().info(
                f"iSDF response: success={res.success} dim={res.dim} "
                f"compute_gradient={res.compute_gradient} n={len(res.signed_distances)} expected={self.N_q}"
            )
            self.get_logger().info(
                f"resp n={len(res.signed_distances)} expected={self.N_q}"
            )
        except Exception as e:
            self.get_logger().error(f"SDF query call failed: {e}")
            return

        if not res.success:
            self.get_logger().warn("SDF query reported success=False")
            return

        n = len(res.signed_distances)
        if n != self.N_q:
            self.get_logger().warn(f"Response length mismatch: got {n}, expected {self.N_q}")
            return

        # --- Extract signed distances into pre-allocated buffers (raw SDF, COARSE)
        sdf = self._sdf_buf
        # Avoid extra large allocation: copy directly from the sequence
        np.copyto(sdf, res.signed_distances, casting="unsafe")

        # Treat sentinel invalid values as missing (optional)
        invalid_sdf_mask = (sdf == self.default_invalid_sdf)

        phi = self._phi_buf
        np.copyto(phi, sdf)
        phi[invalid_sdf_mask] = np.nan

        # --- Coarse grid reshape/unflip
        phi_grid_c = phi.reshape(self.Ny_q, self.Nx_q)
        phi_grid_c = np.flipud(np.fliplr(phi_grid_c)).astype(np.float32, copy=False)

        # --- Upsample coarse -> full using nearest-neighbor repeat
        s = self.query_stride
        if s > 1:
            phi_full = np.repeat(np.repeat(phi_grid_c, s, axis=0), s, axis=1)
            phi_grid = phi_full[:self.Ny, :self.Nx]  # crop to full size
        else:
            phi_grid = phi_grid_c  # already full

        # --- Fill NaNs if requested
        if self.fill_missing_flag and not np.all(np.isfinite(phi_grid)):
            phi_grid = fill_missing(phi_grid)

        # Re-stamp sentinel on invalid cells (upsample the mask consistently)
        if np.any(invalid_sdf_mask):
            mask_c = invalid_sdf_mask.reshape(self.Ny_q, self.Nx_q)
            if s > 1:
                mask_full = np.repeat(np.repeat(mask_c, s, axis=0), s, axis=1)[:self.Ny, :self.Nx]
            else:
                mask_full = mask_c
            phi_grid[mask_full] = self.default_invalid_sdf

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

            if not self.first_info_sent:
                self._publish_info_once()
                self.first_info_sent = True

            np.save(self.sdf_path, phi_out)
            self.get_logger().info(f"Published VF {self.Nx}x{self.Ny}")

        else:  # file mode
            np.save(self.sdf_path, phi_out)
            b = Bool(); b.data = True
            self.vf_pub.publish(b)
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
            "note": f"VF = raw signed distance from iSDF service. query_stride={self.query_stride} (nearest-neighbor upsample).",
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
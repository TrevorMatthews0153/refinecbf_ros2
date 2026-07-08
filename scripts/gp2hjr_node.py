#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from example_interfaces.msg import Bool
from refinecbf_ros2.msg import ValueFunctionMsg

try:
    from scipy import ndimage as _ndimg
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# -------------------------
# Helpers
# -------------------------

def _fields_to_dtype(fields, is_bigendian):
    np_dtype_map = {
        PointField.INT8:    np.int8,
        PointField.UINT8:   np.uint8,
        PointField.INT16:   np.int16,
        PointField.UINT16:  np.uint16,
        PointField.INT32:   np.int32,
        PointField.UINT32:  np.uint32,
        PointField.FLOAT32: np.float32,
        PointField.FLOAT64: np.float64,
    }
    endian = ">" if is_bigendian else "<"

    offs = [f.offset for f in fields]
    order = np.argsort(offs)
    names, formats, offsets = [], [], []
    for idx in order:
        f = fields[idx]
        base = np_dtype_map.get(f.datatype)
        if base is None:
            raise ValueError(f"Unsupported datatype {f.datatype} for field '{f.name}'")
        names.append(f.name)
        formats.append(endian + base().dtype.str[1:])
        offsets.append(f.offset)

    itemsize = max(offsets) + np.dtype(formats[names.index(fields[order[-1]].name)]).itemsize
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": itemsize})


def pointcloud2_to_numpy(msg: PointCloud2, wanted=("x", "y", "sdf", "var_sdf", "gradient_x", "gradient_y")):
    dtype = _fields_to_dtype(msg.fields, msg.is_bigendian)
    npts = msg.width * msg.height
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if len(buf) < msg.row_step * msg.height:
        raise ValueError("PointCloud2 data buffer smaller than expected")

    pts_bytes = buf.reshape((msg.height, msg.row_step))[:, : msg.width * msg.point_step]
    pts_bytes = pts_bytes.reshape((npts, msg.point_step))

    if dtype.itemsize > msg.point_step:
        raise ValueError("Structured dtype larger than point_step (field offsets inconsistent)")

    rec = pts_bytes[:, : dtype.itemsize].view(dtype)
    out = {}
    # for k in wanted:
    #     out[k] = rec[k].astype(np.float32, copy=False) if k in rec.dtype.names else None
    for k in wanted:
        if k in rec.dtype.names:
            arr = np.asarray(rec[k], dtype=np.float32)
            if k == "var_sdf" and arr.ndim == 2:  
                out[k] = arr[:, 0]   # take only the first element → shape (npts,)
            else:
                out[k] = arr
        else:
            out[k] = None

    return out



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


def _aggregate_to_grid_2d(x, y, val, xs, ys, fill_missing=True):
    Nx, Ny = xs.size, ys.size
    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = ys[0], ys[-1]
    dx = (xmax - xmin) / (Nx - 1) if Nx > 1 else 1.0
    dy = (ymax - ymin) / (Ny - 1) if Ny > 1 else 1.0

    m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & np.isfinite(val)
    x, y, val = x[m], y[m], val[m]
    grid = np.full((Ny, Nx), np.nan, dtype=np.float32)
    if x.size == 0:
        return grid, np.zeros_like(grid, dtype=bool)

    ix = np.clip(np.round((x - xmin) / dx).astype(np.int64), 0, Nx - 1)
    iy = np.clip(np.round((y - ymin) / dy).astype(np.int64), 0, Ny - 1)

    lin = iy * Nx + ix
    max_lin = Ny * Nx
    counts = np.bincount(lin, minlength=max_lin)
    sums = np.bincount(lin, weights=val, minlength=max_lin)
    means = np.zeros_like(sums, dtype=np.float32)
    nz = counts > 0
    means[nz] = (sums[nz] / counts[nz]).astype(np.float32)

    grid = means.reshape((Ny, Nx))
    valid = counts.reshape((Ny, Nx)) > 0

    if fill_missing and not np.all(valid):
        if _HAVE_SCIPY:
            nearest_idx = _ndimg.distance_transform_edt(~valid, return_distances=False, return_indices=True)
            grid = grid[tuple(nearest_idx)]
            valid = np.ones_like(valid, dtype=bool)
        else:
            g, v = grid.copy(), valid.copy()
            for _ in range(5):
                g_pad = np.pad(g, 1, mode='edge')
                v_pad = np.pad(v, 1, mode='constant', constant_values=False)
                neigh = [
                    (g_pad[:-2, 1:-1], v_pad[:-2, 1:-1]),
                    (g_pad[2:, 1:-1],  v_pad[2:, 1:-1]),
                    (g_pad[1:-1, :-2], v_pad[1:-1, :-2]),
                    (g_pad[1:-1, 2:],  v_pad[1:-1, 2:]),
                ]
                num = np.zeros_like(g); den = np.zeros_like(g, dtype=np.int32)
                for c, msk in neigh:
                    num[msk] += c[msk]
                    den[msk] += msk[msk].astype(np.int32)
                fill_mask = (~v) & (den > 0)
                g[fill_mask] = (num[fill_mask] / den[fill_mask]).astype(np.float32)
                v[fill_mask] = True
                if np.all(v): break
            grid, valid = g, v

    return grid.astype(np.float32), valid


# -------------------------
# Node
# -------------------------

class SDFPointCloudToGridNode(Node):
    def __init__(self):
        super().__init__("sdf_pointcloud_to_grid")

        # --- Parameters
        # Check if the params are provided in a launch file
        self.declare_parameter("env_config_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/env.yaml")
        self.declare_parameter("mode", "file")  # "pubsub" or "file"
        self.declare_parameter("output_topic", "/env/sdf_update")
        self.declare_parameter("output_topic_grad_x", "/env/sdf_grad_x_update")
        self.declare_parameter("output_topic_grad_y", "/env/sdf_grad_y_update")
        self.declare_parameter("grid_info_topic", "topics/sdf_grid_info")
        self.declare_parameter("sdf_file_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/sdf_sim.npy")
        self.declare_parameter("grad_x_file_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/curr_grad_x.npy")
        self.declare_parameter("grad_y_file_path", "/root/ros2_ws/src/refinecbf_ros2/config/turtlebot/exp5/curr_grad_y.npy")
        self.declare_parameter("fill_missing", True)

        # --- Save Paths
        self.sdf_path =  self.get_parameter("sdf_file_path").get_parameter_value().string_value
        self.gx_path = self.get_parameter("grad_x_file_path").get_parameter_value().string_value
        self.gy_path =  self.get_parameter("grad_y_file_path").get_parameter_value().string_value
        # --- Load env config
        env_path = self.get_parameter("env_config_path").get_parameter_value().string_value
        try:
            dom = parse_env_yaml(env_path)
        except Exception as e:
            self.get_logger().error(f"Failed to read env config '{env_path}': {e}")
            raise

        self.xs = dom["axes"][0]  # x grid (length nx)
        self.ys = dom["axes"][1]  # y grid (length ny)
        self.fill_missing = bool(self.get_parameter("fill_missing").value)

        # --- Publishers
        self.mode = self.get_parameter("mode").get_parameter_value().string_value.lower()
        out_vf_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        out_gx_topic = self.get_parameter("output_topic_grad_x").get_parameter_value().string_value
        out_gy_topic = self.get_parameter("output_topic_grad_y").get_parameter_value().string_value
        info_topic = self.get_parameter("grid_info_topic").get_parameter_value().string_value

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
            self.first_info_sent = False
            self.get_logger().info(
                f"Pub mode: VF->{out_vf_topic}, gradX->{out_gx_topic}, gradY->{out_gy_topic}"
            )
        elif self.mode == "file":
            self.vf_pub = self.create_publisher(Bool, out_vf_topic, qos)
            self.gx_pub = self.create_publisher(Bool, out_gx_topic, qos)
            self.gy_pub = self.create_publisher(Bool, out_gy_topic, qos)
            self.info_pub = None
            self.get_logger().info(
                f"File mode: will publish example_interfaces/Bool(True) on {out_vf_topic}, {out_gx_topic}, {out_gy_topic}"
            )
        else:
            self.get_logger().warn(f"Unknown mode '{self.mode}', defaulting to 'pubsub'")
            self.mode = "pubsub"
            self.vf_pub = self.create_publisher(ValueFunctionMsg, out_vf_topic, qos)
            self.gx_pub = self.create_publisher(ValueFunctionMsg, out_gx_topic, qos)
            self.gy_pub = self.create_publisher(ValueFunctionMsg, out_gy_topic, qos)
            self.info_pub = self.create_publisher(String, info_topic, qos)

        # --- Subscriber
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # self.sub = self.create_subscription(PointCloud2, "/sdf_point_cloud", self._on_cloud, sub_qos)
        self.sub = self.create_subscription(PointCloud2, "/sdf_point_cloud", self._on_cloud, qos_profile_sensor_data)

        self.get_logger().info(
            f"Ready. Mode={self.mode} grid=({self.ys.size}x{self.xs.size}) "
            f"x∈[{self.xs[0]:.2f},{self.xs[-1]:.2f}] y∈[{self.ys[0]:.2f},{self.ys[-1]:.2f}]"
        )

    def _publish_info(self, frame_id: str):
        if not self.info_pub:
            return
        meta = {
            "frame_id": frame_id,
            "xmin": float(self.xs[0]), "xmax": float(self.xs[-1]),
            "ymin": float(self.ys[0]), "ymax": float(self.ys[-1]),
            "nx": int(self.xs.size), "ny": int(self.ys.size),
            "dx": float(self.xs[1] - self.xs[0]) if self.xs.size > 1 else float("nan"),
            "dy": float(self.ys[1] - self.ys[0]) if self.ys.size > 1 else float("nan"),
            "source": "/sdf_point_cloud",
            "note": "Row-major (y,x). VF is SDF + var_sdf; gradients are of VF."
        }
        self.info_pub.publish(String(data=json.dumps(meta)))

    def _publish_bool_triplet(self):
        b = Bool(); b.data = True
        self.vf_pub.publish(b)
        self.gx_pub.publish(b)
        self.gy_pub.publish(b)

    def _on_cloud(self, msg: PointCloud2):
        try:
            arr = pointcloud2_to_numpy(msg, wanted=("x","y","sdf","var_sdf","gradient_x","gradient_y"))
            x = arr["x"]; y = arr["y"]; sdf = arr["sdf"]; var_sdf = arr["var_sdf"]
            gx_raw = arr["gradient_x"]; gy_raw = arr["gradient_y"]

            if x is None or y is None or sdf is None:
                raise ValueError("PointCloud2 missing required fields x,y,sdf")

            x = np.asarray(x).reshape(-1)
            y = np.asarray(y).reshape(-1)
            sdf = np.asarray(sdf).reshape(-1)
            var_sdf = np.zeros_like(sdf, dtype=np.float32) if var_sdf is None else np.asarray(var_sdf).reshape(-1)

            # Augmented scalar field published as the "SDF update"
            phi = sdf - var_sdf

            # Grid the augmented field
            grid_phi, valid = _aggregate_to_grid_2d(x, y, phi, self.xs, self.ys, fill_missing=self.fill_missing)

            # Gradients: prefer the ones from the cloud; grid them directly
            if gx_raw is not None and gy_raw is not None:
                gx_grid, _ = _aggregate_to_grid_2d(x, y, np.asarray(gx_raw).reshape(-1),
                                                self.xs, self.ys, fill_missing=self.fill_missing)
                gy_grid, _ = _aggregate_to_grid_2d(x, y, np.asarray(gy_raw).reshape(-1),
                                                self.xs, self.ys, fill_missing=self.fill_missing)
            else:
                # Fallback: compute grad of the gridded augmented field to cover empty cells
                dy = float(self.ys[1] - self.ys[0]) if self.ys.size > 1 else 1.0
                dx = float(self.xs[1] - self.xs[0]) if self.xs.size > 1 else 1.0
                dVy, dVx = np.gradient(grid_phi, dy, dx, edge_order=2)
                gx_grid = dVx.astype(np.float32)
                gy_grid = dVy.astype(np.float32)

            if self.mode == "pubsub":
                vf_msg = ValueFunctionMsg(); vf_msg.vf = (grid_phi.T).ravel().tolist()
                gx_msg = ValueFunctionMsg(); gx_msg.vf = (gx_grid.T).ravel().tolist()
                gy_msg = ValueFunctionMsg(); gy_msg.vf = (gy_grid.T).ravel().tolist()

                if self.first_info_sent is False:
                    np.save(self.sdf_path, grid_phi.T)
                    np.save(self.gx_path, gx_grid.T)
                    np.save(self.gy_path, gy_grid.T)
                    self.first_info_sent = True
                
                self.vf_pub.publish(vf_msg)
                self.gx_pub.publish(gx_msg)
                self.gy_pub.publish(gy_msg)
                self._publish_info(msg.header.frame_id)

                self.get_logger().info(
                    f"Published VF(SDF+var) {grid_phi.shape[1]}x{grid_phi.shape[0]}, "
                    f"valid={valid.mean()*100:.1f}% | grads published (cloud{' fallback' if gx_raw is None else ''})"
                )
            else:
                self._publish_bool_triplet()
                np.save(self.sdf_path, grid_phi.T)
                np.save(self.gx_path, gx_grid.T)
                np.save(self.gy_path, gy_grid.T)
                self.get_logger().info(
                    f"Saved VF(SDF+var) {grid_phi.shape[1]}x{grid_phi.shape[0]}, "
                    f"valid={valid.mean()*100:.1f}% | grads saved (cloud{' fallback' if gx_raw is None else ''})"
                )

        except Exception as e:
            self.get_logger().error(f"Failed to process PointCloud2: {e}", throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = SDFPointCloudToGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSDurabilityPolicy


@dataclass
class TraversabilityCloud:
    xyz: np.ndarray
    traversability: np.ndarray


class TraversabilityMapNode(Node):
    def __init__(self) -> None:
        super().__init__("traversability_map_node")

        self.declare_parameter("input_cloud_topic", "/semantics/cloud")
        self.declare_parameter("output_map_topic", "/semantics/map/traversability")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("min_points_per_cell", 1)
        self.declare_parameter("publish_on_update", True)
        self.declare_parameter("verbose", True)

        input_cloud_topic = str(self.get_parameter("input_cloud_topic").value)
        output_map_topic = str(self.get_parameter("output_map_topic").value)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.resolution = float(self.get_parameter("resolution").value)
        self.min_points_per_cell = max(1, int(self.get_parameter("min_points_per_cell").value))
        self.publish_on_update = bool(self.get_parameter("publish_on_update").value)
        self.verbose = bool(self.get_parameter("verbose").value)

        if self.resolution <= 0.0:
            raise ValueError("'resolution' debe ser > 0.")

        self.latest_cloud: Optional[TraversabilityCloud] = None

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            input_cloud_topic,
            self.cloud_callback,
            10,
        )
        qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, output_map_topic, qos_profile)

        self.get_logger().info("traversability_map_node iniciado")
        self.get_logger().info(f"  input_cloud_topic   : {input_cloud_topic}")
        self.get_logger().info(f"  output_map_topic    : {output_map_topic}")
        self.get_logger().info(f"  map_frame           : {self.map_frame}")
        self.get_logger().info(f"  resolution          : {self.resolution}")
        self.get_logger().info(f"  min_points_per_cell : {self.min_points_per_cell}")

    def cloud_callback(self, msg: PointCloud2) -> None:
        cloud = self.extract_cloud(msg)
        if cloud is None:
            return

        self.latest_cloud = cloud
        if self.publish_on_update:
            self.publish_map(msg.header)

    def extract_cloud(self, msg: PointCloud2) -> Optional[TraversabilityCloud]:
        try:
            pts_iter = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z", "traversability"),
                skip_nans=True,
            )
            pts = np.array(list(pts_iter))
        except Exception as exc:
            self.get_logger().error(f"No pude leer la nube semántica: {exc}")
            return None

        if pts.size == 0:
            if self.verbose:
                self.get_logger().warning("Nube vacía; no se publica mapa de traversabilidad.")
            return None

        if hasattr(pts.dtype, "names") and pts.dtype.names is not None:
            xyz = np.column_stack(
                [
                    np.asarray(pts["x"], dtype=np.float32),
                    np.asarray(pts["y"], dtype=np.float32),
                    np.asarray(pts["z"], dtype=np.float32),
                ]
            )
            traversability = np.asarray(pts["traversability"], dtype=np.float32)
        else:
            pts = np.asarray(pts, dtype=np.float32)
            xyz = pts[:, :3]
            traversability = pts[:, 3]

        valid = np.isfinite(traversability)
        if not np.any(valid):
            if self.verbose:
                self.get_logger().warning(
                    "La nube recibida solo contiene traversability desconocida; no se publica mapa."
                )
            return None

        xyz = xyz[valid]
        traversability = np.clip(traversability[valid], 0.0, 1.0)
        return TraversabilityCloud(xyz=xyz, traversability=traversability)

    def publish_map(self, source_header: Header) -> None:
        if self.latest_cloud is None or self.latest_cloud.xyz.shape[0] == 0:
            return

        occupancy_grid = self.build_occupancy_grid(self.latest_cloud, source_header)
        self.map_pub.publish(occupancy_grid)

        if self.verbose:
            self.get_logger().info(
                f"Mapa de traversabilidad publicado: "
                f"{occupancy_grid.info.width}x{occupancy_grid.info.height}"
            )

    def build_occupancy_grid(
        self,
        cloud: TraversabilityCloud,
        source_header: Header,
    ) -> OccupancyGrid:
        xy = cloud.xyz[:, :2]
        min_xy = np.min(xy, axis=0)
        max_xy = np.max(xy, axis=0)

        width = max(1, int(np.floor((max_xy[0] - min_xy[0]) / self.resolution)) + 1)
        height = max(1, int(np.floor((max_xy[1] - min_xy[1]) / self.resolution)) + 1)

        cell_x = np.floor((xy[:, 0] - min_xy[0]) / self.resolution).astype(np.int32)
        cell_y = np.floor((xy[:, 1] - min_xy[1]) / self.resolution).astype(np.int32)
        flat_idx = cell_y * width + cell_x

        sum_grid = np.zeros((height * width,), dtype=np.float64)
        count_grid = np.zeros((height * width,), dtype=np.int32)

        np.add.at(sum_grid, flat_idx, cloud.traversability.astype(np.float64))
        np.add.at(count_grid, flat_idx, 1)

        data = np.full((height * width,), -1, dtype=np.int8)
        valid = count_grid >= self.min_points_per_cell
        if np.any(valid):
            mean_traversability = np.zeros_like(sum_grid, dtype=np.float64)
            mean_traversability[valid] = sum_grid[valid] / count_grid[valid]
            occupancy_values = np.rint((1.0 - mean_traversability[valid]) * 100.0).astype(np.int16)
            occupancy_values = np.clip(occupancy_values, 0, 100)
            data[valid] = occupancy_values.astype(np.int8)

        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.map_frame or source_header.frame_id

        grid.info = MapMetaData()
        grid.info.map_load_time = grid.header.stamp
        grid.info.resolution = float(self.resolution)
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = float(min_xy[0])
        grid.info.origin.position.y = float(min_xy[1])
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0

        grid.data = data.tolist()
        return grid


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TraversabilityMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

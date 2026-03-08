#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from geometry_msgs.msg import Pose, Transform
from rtabmap_msgs.msg import MapData, Node as RtabmapNode


PointXYZ = Tuple[float, float, float]
PointXYZRGB = Tuple[float, float, float, int]


class CloudMapProjector(Node):
    def __init__(self) -> None:
        super().__init__("cloud_map_projector")

        self.declare_parameter("map_data_topic", "/rtabmap/mapData")
        self.declare_parameter("cloud_map_topic", "/rtabmap/cloud_map")
        self.declare_parameter("output_cloud_topic", "/semantic_cloud")

        self.declare_parameter("target_frame", "map")
        self.declare_parameter("pixel_step", 1)
        self.declare_parameter("max_depth_m", 20.0)
        self.declare_parameter("z_buffer_margin", 0.02)  # m
        self.declare_parameter("verbose", True)

        map_data_topic = self.get_parameter("map_data_topic").value
        cloud_map_topic = self.get_parameter("cloud_map_topic").value
        output_cloud_topic = self.get_parameter("output_cloud_topic").value

        self.target_frame = self.get_parameter("target_frame").value
        self.pixel_step = int(self.get_parameter("pixel_step").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.z_buffer_margin = float(self.get_parameter("z_buffer_margin").value)
        self.verbose = bool(self.get_parameter("verbose").value)

        self.bridge = CvBridge()

        self.latest_cloud_points: Optional[np.ndarray] = None  # Nx3 float32 in map
        self.latest_cloud_stamp = None

        self.graph_poses: Dict[int, Pose] = {}
        self.processed_node_ids: set[int] = set()

        self.map_data_sub = self.create_subscription(
            MapData,
            map_data_topic,
            self.map_data_callback,
            10,
        )
        self.cloud_map_sub = self.create_subscription(
            PointCloud2,
            cloud_map_topic,
            self.cloud_map_callback,
            10,
        )

        self.cloud_pub = self.create_publisher(PointCloud2, output_cloud_topic, 10)

        self.get_logger().info("cloud_map_projector iniciado")
        self.get_logger().info(f"  map_data_topic    : {map_data_topic}")
        self.get_logger().info(f"  cloud_map_topic   : {cloud_map_topic}")
        self.get_logger().info(f"  output_cloud_topic: {output_cloud_topic}")

    # =========================================================
    # Callbacks
    # =========================================================

    def cloud_map_callback(self, msg: PointCloud2) -> None:
        try:
            pts = list(
                point_cloud2.read_points(
                    msg,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            )
            if not pts:
                return

            self.latest_cloud_points = np.asarray(pts, dtype=np.float32)
            self.latest_cloud_stamp = msg.header.stamp

            if self.verbose:
                self.get_logger().info(
                    f"cloud_map recibido con {self.latest_cloud_points.shape[0]} puntos"
                )
        except Exception as e:
            self.get_logger().error(f"Error leyendo cloud_map: {e}")

    def map_data_callback(self, msg: MapData) -> None:
        self.update_graph_poses(msg)

        if self.latest_cloud_points is None or self.latest_cloud_points.shape[0] == 0:
            return

        # Procesar nodos nuevos, el más reciente primero
        new_nodes = [n for n in msg.nodes if int(n.id) not in self.processed_node_ids]
        if not new_nodes:
            return

        # Intentamos del más reciente al más antiguo
        for node in reversed(new_nodes):
            ok = self.process_node_on_cloud(node)
            self.processed_node_ids.add(int(node.id))
            if ok:
                break

    # =========================================================
    # Graph poses
    # =========================================================

    def update_graph_poses(self, msg: MapData) -> None:
        graph = msg.graph
        if hasattr(graph, "poses_id") and hasattr(graph, "poses"):
            for node_id, pose in zip(graph.poses_id, graph.poses):
                self.graph_poses[int(node_id)] = pose

    # =========================================================
    # Main processing
    # =========================================================

    def process_node_on_cloud(self, node: RtabmapNode) -> bool:
        node_id = int(node.id)

        if node_id not in self.graph_poses:
            if self.verbose:
                self.get_logger().warning(f"Node {node_id}: sin pose optimizada en graph")
            return False

        if not hasattr(node, "data"):
            return False

        data = node.data

        if len(data.left_camera_info) == 0 or len(data.local_transform) == 0:
            if self.verbose:
                self.get_logger().warning(
                    f"Node {node_id}: faltan left_camera_info o local_transform"
                )
            return False

        rgb = self.extract_rgb(data.left, data.left_compressed)
        if rgb is None:
            if self.verbose:
                self.get_logger().warning(f"Node {node_id}: no pude decodificar RGB")
            return False

        cam_info = data.left_camera_info[0]
        pose_map_base = self.graph_poses[node_id]
        tf_base_camera = data.local_transform[0]

        # map -> camera
        T_map_base = self.pose_to_matrix(pose_map_base)
        T_base_camera = self.transform_to_matrix(tf_base_camera)
        T_map_camera = T_map_base @ T_base_camera
        T_camera_map = np.linalg.inv(T_map_camera)

        colored_points = self.project_cloud_with_zbuffer(
            cloud_map_xyz=self.latest_cloud_points,
            rgb=rgb,
            camera_info=cam_info,
            T_camera_map=T_camera_map,
        )

        if not colored_points:
            self.get_logger().warning(f"Node {node_id}: no hubo puntos visibles para colorear")
            return False

        cloud_msg = self.create_cloud_msg(colored_points, self.target_frame)
        self.cloud_pub.publish(cloud_msg)

        self.get_logger().info(
            f"Node {node_id}: publicados {len(colored_points)} puntos coloreados"
        )
        return True

    # =========================================================
    # Projection with z-buffer
    # =========================================================

    def project_cloud_with_zbuffer(
        self,
        cloud_map_xyz: np.ndarray,
        rgb: np.ndarray,
        camera_info: CameraInfo,
        T_camera_map: np.ndarray,
    ) -> List[PointXYZRGB]:
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        h, w = rgb.shape[:2]

        # Transformar todos los puntos map -> camera
        pts_map_h = np.hstack(
            [cloud_map_xyz.astype(np.float64), np.ones((cloud_map_xyz.shape[0], 1), dtype=np.float64)]
        )
        pts_cam_h = (T_camera_map @ pts_map_h.T).T
        pts_cam = pts_cam_h[:, :3]

        z = pts_cam[:, 2]
        valid = (z > 0.0) & (z < self.max_depth_m)
        if not np.any(valid):
            return []

        pts_cam = pts_cam[valid]
        pts_map = cloud_map_xyz[valid]

        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = pts_cam[:, 2]

        u = np.round((fx * x / z) + cx).astype(np.int32)
        v = np.round((fy * y / z) + cy).astype(np.int32)

        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(inside):
            return []

        pts_cam = pts_cam[inside]
        pts_map = pts_map[inside]
        u = u[inside]
        v = v[inside]
        z = z[inside]

        # z-buffer por píxel: índice del punto más cercano
        depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
        index_buffer = np.full((h, w), -1, dtype=np.int32)

        for i in range(0, len(z), self.pixel_step):
            uu = u[i]
            vv = v[i]
            zz = z[i]
            if zz < depth_buffer[vv, uu]:
                depth_buffer[vv, uu] = zz
                index_buffer[vv, uu] = i

        colored_points: List[PointXYZRGB] = []

        ys, xs = np.where(index_buffer >= 0)
        for vv, uu in zip(ys, xs):
            i = index_buffer[vv, uu]
            zz = z[i]
            if zz > depth_buffer[vv, uu] + self.z_buffer_margin:
                continue

            px = pts_map[i, 0]
            py = pts_map[i, 1]
            pz = pts_map[i, 2]

            b, g, r = rgb[vv, uu]
            rgb_uint32 = self.pack_rgb(int(r), int(g), int(b))
            colored_points.append((float(px), float(py), float(pz), rgb_uint32))

        return colored_points

    # =========================================================
    # RGB extraction
    # =========================================================

    def extract_rgb(self, raw_msg: Image, compressed_data) -> Optional[np.ndarray]:
        img = self.extract_rgb_raw(raw_msg)
        if img is not None:
            return img
        return self.extract_rgb_compressed(compressed_data)

    def extract_rgb_raw(self, msg: Image) -> Optional[np.ndarray]:
        if msg.height == 0 or msg.width == 0 or len(msg.data) == 0:
            return None
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return None

    def extract_rgb_compressed(self, compressed_data) -> Optional[np.ndarray]:
        arr = self.to_uint8_array(compressed_data)
        if arr is None or arr.size == 0:
            return None
        try:
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def to_uint8_array(self, data) -> Optional[np.ndarray]:
        try:
            if data is None:
                return None
            if isinstance(data, (bytes, bytearray)):
                return np.frombuffer(data, dtype=np.uint8)
            return np.asarray(data, dtype=np.uint8)
        except Exception:
            return None

    # =========================================================
    # Transforms
    # =========================================================

    def pose_to_matrix(self, pose: Pose) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.quaternion_to_rotation_matrix(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return T

    def transform_to_matrix(self, tf_msg: Transform) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.quaternion_to_rotation_matrix(
            tf_msg.rotation.x,
            tf_msg.rotation.y,
            tf_msg.rotation.z,
            tf_msg.rotation.w,
        )
        T[:3, 3] = [tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z]
        return T

    def quaternion_to_rotation_matrix(
        self,
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> np.ndarray:
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    # =========================================================
    # PointCloud2 output
    # =========================================================

    def pack_rgb(self, r: int, g: int, b: int) -> int:
        return (r << 16) | (g << 8) | b

    def create_cloud_msg(
        self,
        points: List[PointXYZRGB],
        frame_id: str,
    ) -> PointCloud2:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        return point_cloud2.create_cloud(header, fields, points)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CloudMapProjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrumpido por teclado.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

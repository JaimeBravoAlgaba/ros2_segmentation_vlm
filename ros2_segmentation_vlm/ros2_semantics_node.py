#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import io
import socket
import struct

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Transform
from rtabmap_msgs.msg import MapData, Node as RtabmapNode
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


VoxelKey = Tuple[int, int, int]
PointXYZRGB = Tuple[float, float, float, int]


@dataclass
class NodeObservation:
    node_id: int
    rgb: np.ndarray
    camera_info: CameraInfo
    local_transform: Transform


class CloudMapProjector(Node):
    def __init__(self) -> None:
        super().__init__("cloud_map_projector")

        self.declare_parameter("map_data_topic", "/rtabmap/mapData")
        self.declare_parameter("cloud_map_topic", "/rtabmap/cloud_map")
        self.declare_parameter("output_cloud_topic", "/semantic_cloud")
        self.declare_parameter("segmentation_server_host", "127.0.0.1")
        self.declare_parameter("segmentation_server_port", 8765)
        self.declare_parameter("segmentation_socket_timeout_sec", 30.0)
        self.declare_parameter("reconnect_delay_sec", 1.0)

        self.declare_parameter("target_frame", "map")
        self.declare_parameter("pixel_step", 1)
        self.declare_parameter("max_depth_m", 20.0)
        self.declare_parameter("z_buffer_margin", 0.02)
        self.declare_parameter("color_voxel_size", 0.02)
        self.declare_parameter("unknown_rgb", [0, 0, 0])
        self.declare_parameter("rebuild_rate_hz", 0.5)
        self.declare_parameter("verbose", True)

        map_data_topic = self.get_parameter("map_data_topic").value
        cloud_map_topic = self.get_parameter("cloud_map_topic").value
        output_cloud_topic = self.get_parameter("output_cloud_topic").value

        self.target_frame = self.get_parameter("target_frame").value
        self.pixel_step = int(self.get_parameter("pixel_step").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.z_buffer_margin = float(self.get_parameter("z_buffer_margin").value)
        self.color_voxel_size = float(self.get_parameter("color_voxel_size").value)
        unknown_rgb_value = self.get_parameter("unknown_rgb").value
        if isinstance(unknown_rgb_value, (list, tuple)) and len(unknown_rgb_value) == 3:
            self.unknown_rgb = tuple(int(np.clip(v, 0, 255)) for v in unknown_rgb_value)
        else:
            self.unknown_rgb = (0, 0, 0)
        self.segmentation_server_host = str(self.get_parameter("segmentation_server_host").value)
        self.segmentation_server_port = int(self.get_parameter("segmentation_server_port").value)
        self.segmentation_socket_timeout_sec = float(
            self.get_parameter("segmentation_socket_timeout_sec").value
        )
        self.reconnect_delay_sec = float(self.get_parameter("reconnect_delay_sec").value)
        rebuild_rate_hz = float(self.get_parameter("rebuild_rate_hz").value)
        self.verbose = bool(self.get_parameter("verbose").value)

        self.bridge = CvBridge()

        # Geometría actual del mapa
        self.latest_cloud_points: Optional[np.ndarray] = None  # Nx3 float32
        self.latest_cloud_stamp = None

        # Grafo optimizado actual
        self.graph_poses: Dict[int, Pose] = {}
        self.last_graph_signature: Optional[Tuple[int, int]] = None
        self.last_map_node_ids_signature: Optional[Tuple[int, int, int]] = None

        # Observaciones por nodo
        self.node_observations: Dict[int, NodeObservation] = {}
        self.seg_sock: Optional[socket.socket] = None

        # Acumulador reconstruido en cada rebuild
        # voxel -> [sum_r, sum_g, sum_b, count]
        self.color_accumulator: Dict[VoxelKey, np.ndarray] = {}

        self.map_dirty = False
        self.last_rebuild_summary = ""

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

        period = 1.0 / rebuild_rate_hz if rebuild_rate_hz > 0.0 else 2.0
        self.rebuild_timer = self.create_timer(period, self.rebuild_if_needed)

        self.get_logger().info("cloud_map_projector iniciado")
        self.get_logger().info(f"  map_data_topic     : {map_data_topic}")
        self.get_logger().info(f"  cloud_map_topic    : {cloud_map_topic}")
        self.get_logger().info(f"  output_cloud_topic : {output_cloud_topic}")
        self.get_logger().info(
            f"  seg_server         : {self.segmentation_server_host}:{self.segmentation_server_port}"
        )
        self.get_logger().info(f"  seg_timeout_sec    : {self.segmentation_socket_timeout_sec}")
        self.get_logger().info(f"  unknown_rgb        : {self.unknown_rgb}")
        self.get_logger().info(f"  color_voxel_size   : {self.color_voxel_size}")
        self.get_logger().info(f"  rebuild_rate_hz    : {rebuild_rate_hz}")

        self._connect_segmentation_server()

    # =========================================================
    # Callbacks
    # =========================================================

    def cloud_map_callback(self, msg: PointCloud2) -> None:
        try:
            pts_iter = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            pts = np.array(list(pts_iter))

            if pts.size == 0:
                return

            if hasattr(pts.dtype, "names") and pts.dtype.names is not None:
                xyz = np.column_stack([pts["x"], pts["y"], pts["z"]])
            else:
                xyz = pts

            self.latest_cloud_points = np.asarray(xyz, dtype=np.float32)
            self.latest_cloud_stamp = msg.header.stamp
            self.map_dirty = True

            if self.verbose:
                self.get_logger().info(
                    f"cloud_map recibido con {self.latest_cloud_points.shape[0]} puntos"
                )
        except Exception as e:
            self.get_logger().error(f"Error leyendo cloud_map: {e}")

    def map_data_callback(self, msg: MapData) -> None:
        self.maybe_reset_state_on_map_reinit(msg)
        self.update_graph_poses(msg)

        graph_signature = self.compute_graph_signature(msg)
        graph_changed = graph_signature != self.last_graph_signature
        self.last_graph_signature = graph_signature

        if graph_changed:
            self.map_dirty = True
            if self.verbose:
                self.get_logger().info("Cambio detectado en el grafo de RTAB-Map")

        candidate_node: Optional[RtabmapNode] = None
        for node in reversed(msg.nodes):
            node_id = int(node.id)
            if node_id not in self.node_observations:
                candidate_node = node
                break

        if candidate_node is None:
            if self.verbose:
                self.get_logger().info(
                    f"Sin nodos nuevos para segmentar. total_msg_nodes={len(msg.nodes)}, "
                    f"total_ready={len(self.node_observations)}"
                )
            return

        obs = self.extract_observation_from_node(candidate_node)
        if obs is None:
            return

        segmented_rgb = self.segment_image_blocking(obs.rgb)
        if segmented_rgb is None:
            if self.verbose:
                self.get_logger().warning(
                    f"Node {obs.node_id}: segmentacion fallida o timeout; no se proyecta."
                )
            return

        obs.rgb = segmented_rgb
        self.node_observations[obs.node_id] = obs
        self.map_dirty = True

    # =========================================================
    # Graph poses
    # =========================================================

    def update_graph_poses(self, msg: MapData) -> None:
        graph = msg.graph

        if hasattr(graph, "poses_id") and hasattr(graph, "poses"):
            for node_id, pose in zip(graph.poses_id, graph.poses):
                self.graph_poses[int(node_id)] = pose
            return

        # fallback defensivo
        if hasattr(graph, "node_ids") and hasattr(graph, "poses"):
            for node_id, pose in zip(graph.node_ids, graph.poses):
                self.graph_poses[int(node_id)] = pose

    # =========================================================
    # Rebuild logic
    # =========================================================

    def rebuild_if_needed(self) -> None:
        if not self.map_dirty:
            return

        if self.latest_cloud_points is None or self.latest_cloud_points.shape[0] == 0:
            return

        if not self.node_observations:
            return

        self.rebuild_full_map()

    def rebuild_full_map(self) -> None:
        self.color_accumulator.clear()

        total_visible = 0
        used_nodes = 0
        skipped_nodes = 0

        node_ids = sorted(self.node_observations.keys())

        for node_id in node_ids:
            obs = self.node_observations[node_id]

            if node_id not in self.graph_poses:
                skipped_nodes += 1
                continue

            pose_map_base = self.graph_poses[node_id]

            T_map_base = self.pose_to_matrix(pose_map_base)
            T_base_camera = self.transform_to_matrix(obs.local_transform)
            T_map_camera = T_map_base @ T_base_camera
            T_camera_map = np.linalg.inv(T_map_camera)

            visible_points = self.project_cloud_with_zbuffer(
                cloud_map_xyz=self.latest_cloud_points,
                rgb=obs.rgb,
                camera_info=obs.camera_info,
                T_camera_map=T_camera_map,
            )

            if visible_points:
                self.accumulate_visible_points(visible_points)
                total_visible += len(visible_points)
                used_nodes += 1
            else:
                skipped_nodes += 1

        self.publish_accumulated_cloud()

        self.last_rebuild_summary = (
            f"Rebuild completo: "
            f"nodes_total={len(node_ids)}, "
            f"nodes_used={used_nodes}, "
            f"nodes_skipped={skipped_nodes}, "
            f"visible_points={total_visible}, "
            f"colored_voxels={len(self.color_accumulator)}"
        )
        self.get_logger().info(self.last_rebuild_summary)

        self.map_dirty = False

    # =========================================================
    # Observation extraction
    # =========================================================

    def extract_observation_from_node(self, node: RtabmapNode) -> Optional[NodeObservation]:
        if not hasattr(node, "data"):
            return None

        data = node.data

        if len(data.left_camera_info) == 0:
            if self.verbose:
                self.get_logger().warning(f"Node {node.id}: sin left_camera_info")
            return None

        if len(data.local_transform) == 0:
            if self.verbose:
                self.get_logger().warning(f"Node {node.id}: sin local_transform")
            return None

        rgb = self.extract_rgb(data.left, data.left_compressed)
        if rgb is None:
            if self.verbose:
                self.get_logger().warning(f"Node {node.id}: no pude decodificar RGB")
            return None

        return NodeObservation(
            node_id=int(node.id),
            rgb=rgb,
            camera_info=data.left_camera_info[0],
            local_transform=data.local_transform[0],
        )

    def _connect_segmentation_server(self) -> None:
        if self.seg_sock is not None:
            return
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.segmentation_socket_timeout_sec)
            s.connect((self.segmentation_server_host, self.segmentation_server_port))
            self.seg_sock = s
            self.get_logger().info(
                f"Conectado al servidor de segmentacion en "
                f"{self.segmentation_server_host}:{self.segmentation_server_port}"
            )
        except Exception as e:
            self.seg_sock = None
            self.get_logger().warning(f"No se pudo conectar al servidor de segmentacion: {e}")

    def _close_segmentation_socket(self) -> None:
        if self.seg_sock is None:
            return
        try:
            self.seg_sock.close()
        except Exception:
            pass
        self.seg_sock = None

    def _recvall(self, sock: socket.socket, n: int) -> Optional[bytes]:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _send_msg(self, sock: socket.socket, payload: bytes) -> None:
        sock.sendall(struct.pack(">I", len(payload)) + payload)

    def _recv_msg(self, sock: socket.socket) -> Optional[bytes]:
        raw_len = self._recvall(sock, 4)
        if raw_len is None:
            return None
        msg_len = struct.unpack(">I", raw_len)[0]
        return self._recvall(sock, msg_len)

    def segment_image_blocking(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.seg_sock is None:
            self._connect_segmentation_server()
        if self.seg_sock is None:
            return None

        try:
            buf = io.BytesIO()
            np.save(buf, bgr, allow_pickle=False)
            self._send_msg(self.seg_sock, buf.getvalue())

            response = self._recv_msg(self.seg_sock)
            if response is None or len(response) < 12:
                raise RuntimeError("respuesta vacia o incompleta")

            h, w, c = struct.unpack(">III", response[:12])
            img_data = response[12:]
            expected_len = h * w * c
            if len(img_data) != expected_len:
                raise RuntimeError(
                    f"tamano inesperado: got={len(img_data)}, expected={expected_len}"
                )

            seg_rgb = np.frombuffer(img_data, dtype=np.uint8).reshape((h, w, c))
            seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
            if seg_bgr.shape[:2] != bgr.shape[:2]:
                seg_bgr = cv2.resize(
                    seg_bgr,
                    (bgr.shape[1], bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            return seg_bgr
        except Exception as e:
            self.get_logger().warning(f"Fallo de segmentacion bloqueante: {e}")
            self._close_segmentation_socket()
            time.sleep(self.reconnect_delay_sec)
            return None

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

        pts_map_h = np.hstack(
            [
                cloud_map_xyz.astype(np.float64),
                np.ones((cloud_map_xyz.shape[0], 1), dtype=np.float64),
            ]
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

        pts_map = pts_map[inside]
        u = u[inside]
        v = v[inside]
        z = z[inside]

        depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
        index_buffer = np.full((h, w), -1, dtype=np.int32)

        for i in range(0, len(z), self.pixel_step):
            uu = u[i]
            vv = v[i]
            zz = z[i]
            if zz < depth_buffer[vv, uu]:
                depth_buffer[vv, uu] = zz
                index_buffer[vv, uu] = i

        visible_points: List[PointXYZRGB] = []

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
            visible_points.append((float(px), float(py), float(pz), rgb_uint32))

        return visible_points

    # =========================================================
    # Color accumulation
    # =========================================================

    def accumulate_visible_points(self, points: List[PointXYZRGB]) -> None:
        for x, y, z, rgb_uint32 in points:
            key = self.xyz_to_voxel_key(x, y, z)

            r = (rgb_uint32 >> 16) & 0xFF
            g = (rgb_uint32 >> 8) & 0xFF
            b = rgb_uint32 & 0xFF

            if key not in self.color_accumulator:
                self.color_accumulator[key] = np.array(
                    [float(r), float(g), float(b), 1.0], dtype=np.float64
                )
            else:
                self.color_accumulator[key][0] += float(r)
                self.color_accumulator[key][1] += float(g)
                self.color_accumulator[key][2] += float(b)
                self.color_accumulator[key][3] += 1.0

    def publish_accumulated_cloud(self) -> None:
        if self.latest_cloud_points is None or self.latest_cloud_points.shape[0] == 0:
            return

        unk_r, unk_g, unk_b = self.unknown_rgb
        output_points: List[PointXYZRGB] = []

        for i in range(self.latest_cloud_points.shape[0]):
            x = float(self.latest_cloud_points[i, 0])
            y = float(self.latest_cloud_points[i, 1])
            z = float(self.latest_cloud_points[i, 2])

            key = self.xyz_to_voxel_key(x, y, z)

            if key in self.color_accumulator:
                acc = self.color_accumulator[key]
                count = max(acc[3], 1.0)
                r = int(np.clip(acc[0] / count, 0, 255))
                g = int(np.clip(acc[1] / count, 0, 255))
                b = int(np.clip(acc[2] / count, 0, 255))
            else:
                r, g, b = unk_r, unk_g, unk_b

            rgb_uint32 = self.pack_rgb(r, g, b)
            output_points.append((x, y, z, rgb_uint32))

        cloud_msg = self.create_cloud_msg(output_points, self.target_frame)
        self.cloud_pub.publish(cloud_msg)

        if self.verbose:
            self.get_logger().info(
                f"Mapa publicado: total_points={len(output_points)}, "
                f"colored_voxels={len(self.color_accumulator)}"
            )

    def xyz_to_voxel_key(self, x: float, y: float, z: float) -> VoxelKey:
        inv = 1.0 / self.color_voxel_size
        return (
            int(np.floor(x * inv)),
            int(np.floor(y * inv)),
            int(np.floor(z * inv)),
        )

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
    # Graph / transforms
    # =========================================================

    def compute_graph_signature(self, msg: MapData) -> Tuple[int, int]:
        num_nodes = len(msg.nodes)
        num_links = -1

        if hasattr(msg.graph, "links"):
            try:
                num_links = len(msg.graph.links)
            except Exception:
                pass
        elif hasattr(msg.graph, "constraints"):
            try:
                num_links = len(msg.graph.constraints)
            except Exception:
                pass

        return (num_nodes, num_links)

    def compute_node_ids_signature(self, msg: MapData) -> Optional[Tuple[int, int, int]]:
        if not msg.nodes:
            return None
        node_ids = [int(n.id) for n in msg.nodes]
        return (len(node_ids), min(node_ids), max(node_ids))

    def maybe_reset_state_on_map_reinit(self, msg: MapData) -> None:
        current_sig = self.compute_node_ids_signature(msg)
        previous_sig = self.last_map_node_ids_signature
        self.last_map_node_ids_signature = current_sig

        if previous_sig is None or current_sig is None:
            return

        prev_count, prev_min_id, prev_max_id = previous_sig
        curr_count, curr_min_id, curr_max_id = current_sig

        # Heurística robusta de reinicio/relocalización fuerte del mapa:
        # 1) caída brusca del número de nodos o
        # 2) rango de IDs retrocede claramente.
        node_count_dropped = curr_count < prev_count and curr_count < max(10, prev_count // 3)
        ids_moved_back = curr_max_id < prev_min_id or curr_min_id < prev_min_id

        if not (node_count_dropped or ids_moved_back):
            return

        self.get_logger().warning(
            "Detectado posible reset/reinicio de mapa RTAB-Map. "
            "Limpiando estado semántico e inferencia en vuelo."
        )
        self.reset_semantic_state()

    def reset_semantic_state(self) -> None:
        self.node_observations.clear()
        self.graph_poses.clear()
        self.color_accumulator.clear()
        self._close_segmentation_socket()
        self.map_dirty = True

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
        node._close_segmentation_socket()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

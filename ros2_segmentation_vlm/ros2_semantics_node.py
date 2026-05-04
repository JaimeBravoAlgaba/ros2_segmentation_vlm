#!/usr/bin/env python3

from dataclasses import dataclass
import io
import os
import socket
import struct
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Transform
from rclpy.node import Node
from rtabmap_msgs.msg import MapData, Node as RtabmapNode
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from ros2_segmentation_vlm.segmentation_protocol import (
    MSG_CONFIGURE_ACK,
    MSG_ERROR,
    MSG_SEGMENT_RESULT,
    decode_message_type,
    encode_configure_request,
    encode_segment_request,
)
from ros2_segmentation_vlm.semantic_classes import (
    SemanticClasses,
    colorize_class_map,
    load_semantic_classes,
)


VoxelKey = Tuple[int, int, int]
ProjectedPoint = Tuple[float, float, float, int]
PointXYZRGBSemantic = Tuple[float, float, float, int, int, int, float, float]


@dataclass
class ExtractedObservation:
    node_id: int
    rgb: np.ndarray
    camera_info: CameraInfo
    local_transform: Transform


@dataclass
class NodeObservation:
    node_id: int
    class_map: np.ndarray
    camera_info: CameraInfo
    local_transform: Transform


class CloudMapProjector(Node):
    def __init__(self) -> None:
        super().__init__("cloud_map_projector")

        default_semantic_path = os.path.join(
            get_package_share_directory("ros2_segmentation_vlm"),
            "config",
            "arena_semantic_classes.json",
        )

        self.declare_parameter("map_data_topic", "/rtabmap/mapData")
        self.declare_parameter("cloud_map_topic", "/rtabmap/cloud_map")
        self.declare_parameter("output_cloud_topic", "/semantics/cloud")
        self.declare_parameter("output_image_topic", "/semantics/image")
        self.declare_parameter("output_class_topic", "/semantics/class")
        self.declare_parameter("segmentation_server_host", "127.0.0.1")
        self.declare_parameter("segmentation_server_port", 8765)
        self.declare_parameter("segmentation_socket_timeout_sec", 30.0)
        self.declare_parameter("reconnect_delay_sec", 1.0)
        self.declare_parameter("semantic_classes_path", default_semantic_path)

        self.declare_parameter("target_frame", "map")
        self.declare_parameter("pixel_step", 1)
        self.declare_parameter("max_depth_m", 20.0)
        self.declare_parameter("z_buffer_margin", 0.02)
        self.declare_parameter("color_voxel_size", 0.02)
        self.declare_parameter("rebuild_rate_hz", 0.5)
        self.declare_parameter("verbose", True)

        map_data_topic = self.get_parameter("map_data_topic").value
        cloud_map_topic = self.get_parameter("cloud_map_topic").value
        output_cloud_topic = self.get_parameter("output_cloud_topic").value
        output_image_topic = self.get_parameter("output_image_topic").value
        output_class_topic = self.get_parameter("output_class_topic").value

        semantic_classes_path = str(self.get_parameter("semantic_classes_path").value)
        self.semantic_classes = load_semantic_classes(semantic_classes_path)
        self.unknown_class_id = self.semantic_classes.unknown_class_id
        self.unknown_rgb = self.semantic_classes.unknown_color

        self.target_frame = self.get_parameter("target_frame").value
        self.pixel_step = int(self.get_parameter("pixel_step").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.z_buffer_margin = float(self.get_parameter("z_buffer_margin").value)
        self.color_voxel_size = float(self.get_parameter("color_voxel_size").value)
        self.segmentation_server_host = str(self.get_parameter("segmentation_server_host").value)
        self.segmentation_server_port = int(self.get_parameter("segmentation_server_port").value)
        self.segmentation_socket_timeout_sec = float(
            self.get_parameter("segmentation_socket_timeout_sec").value
        )
        self.reconnect_delay_sec = float(self.get_parameter("reconnect_delay_sec").value)
        rebuild_rate_hz = float(self.get_parameter("rebuild_rate_hz").value)
        self.verbose = bool(self.get_parameter("verbose").value)

        self.bridge = CvBridge()

        self.latest_cloud_points: Optional[np.ndarray] = None
        self.latest_cloud_stamp = None

        self.graph_poses: Dict[int, Pose] = {}
        self.last_graph_signature: Optional[Tuple[int, int]] = None
        self.last_map_node_ids_signature: Optional[Tuple[int, int, int]] = None

        self.node_observations: Dict[int, NodeObservation] = {}
        self.seg_sock: Optional[socket.socket] = None

        self.class_votes: Dict[VoxelKey, Dict[int, int]] = {}
        # Punto de extensión para futuros atributos por clase, p. ej. coste o transitabilidad.
        self.voxel_attribute_votes: Dict[str, Dict[VoxelKey, Dict[int, int]]] = {}

        self.map_dirty = False
        self.last_rebuild_summary = ""

        self.map_data_sub = self.create_subscription(MapData, map_data_topic, self.map_data_callback, 10)
        self.cloud_map_sub = self.create_subscription(
            PointCloud2,
            cloud_map_topic,
            self.cloud_map_callback,
            10,
        )
        self.cloud_pub = self.create_publisher(PointCloud2, output_cloud_topic, 10)
        self.image_pub = self.create_publisher(Image, output_image_topic, 10)
        self.class_pub = self.create_publisher(Image, output_class_topic, 10)

        period = 1.0 / rebuild_rate_hz if rebuild_rate_hz > 0.0 else 2.0
        self.rebuild_timer = self.create_timer(period, self.rebuild_if_needed)

        self.get_logger().info("cloud_map_projector iniciado")
        self.get_logger().info(f"  semantic_classes    : {self.semantic_classes.source_path}")
        self.get_logger().info(f"  semantic_prompts    : {len(self.semantic_classes.class_names)}")
        self.get_logger().info(f"  unknown_class_id    : {self.unknown_class_id}")
        self.get_logger().info(f"  unknown_rgb         : {self.unknown_rgb}")
        self.get_logger().info(f"  seg_server          : {self.segmentation_server_host}:{self.segmentation_server_port}")
        self.get_logger().info(f"  seg_timeout_sec     : {self.segmentation_socket_timeout_sec}")
        self.get_logger().info(f"  color_voxel_size    : {self.color_voxel_size}")
        self.get_logger().info(f"  rebuild_rate_hz     : {rebuild_rate_hz}")

        self._connect_segmentation_server()

    def cloud_map_callback(self, msg: PointCloud2) -> None:
        try:
            pts_iter = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
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
        except Exception as exc:
            self.get_logger().error(f"Error leyendo cloud_map: {exc}")

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

        class_map = self.segment_image_blocking(obs.rgb)
        if class_map is None:
            if self.verbose:
                self.get_logger().warning(
                    f"Node {obs.node_id}: segmentacion fallida o timeout; no se proyecta."
                )
            return

        segmented_rgb = colorize_class_map(class_map, self.semantic_classes)
        segmented_bgr = cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(segmented_bgr, encoding="bgr8"))
        self.class_pub.publish(self.bridge.cv2_to_imgmsg(class_map, encoding="mono8"))

        self.node_observations[obs.node_id] = NodeObservation(
            node_id=obs.node_id,
            class_map=class_map,
            camera_info=obs.camera_info,
            local_transform=obs.local_transform,
        )
        self.map_dirty = True

    def update_graph_poses(self, msg: MapData) -> None:
        graph = msg.graph

        if hasattr(graph, "poses_id") and hasattr(graph, "poses"):
            for node_id, pose in zip(graph.poses_id, graph.poses):
                self.graph_poses[int(node_id)] = pose
            return

        if hasattr(graph, "node_ids") and hasattr(graph, "poses"):
            for node_id, pose in zip(graph.node_ids, graph.poses):
                self.graph_poses[int(node_id)] = pose

    def rebuild_if_needed(self) -> None:
        if not self.map_dirty:
            return
        if self.latest_cloud_points is None or self.latest_cloud_points.shape[0] == 0:
            return
        if not self.node_observations:
            return

        self.rebuild_full_map()

    def rebuild_full_map(self) -> None:
        self.class_votes.clear()

        total_visible = 0
        used_nodes = 0
        skipped_nodes = 0

        for node_id in sorted(self.node_observations.keys()):
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
                class_map=obs.class_map,
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
            f"nodes_total={len(self.node_observations)}, "
            f"nodes_used={used_nodes}, "
            f"nodes_skipped={skipped_nodes}, "
            f"visible_points={total_visible}, "
            f"classified_voxels={len(self.class_votes)}"
        )
        self.get_logger().info(self.last_rebuild_summary)
        self.map_dirty = False

    def extract_observation_from_node(self, node: RtabmapNode) -> Optional[ExtractedObservation]:
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

        return ExtractedObservation(
            node_id=int(node.id),
            rgb=rgb,
            camera_info=data.left_camera_info[0],
            local_transform=data.local_transform[0],
        )

    def _connect_segmentation_server(self) -> None:
        if self.seg_sock is not None:
            return

        try:
            seg_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            seg_sock.settimeout(self.segmentation_socket_timeout_sec)
            seg_sock.connect((self.segmentation_server_host, self.segmentation_server_port))
            self.seg_sock = seg_sock
            self._configure_segmentation_server()
            self.get_logger().info(
                f"Conectado al servidor de segmentacion en "
                f"{self.segmentation_server_host}:{self.segmentation_server_port}"
            )
        except Exception as exc:
            self.seg_sock = None
            self.get_logger().warning(f"No se pudo conectar al servidor de segmentacion: {exc}")

    def _configure_segmentation_server(self) -> None:
        if self.seg_sock is None:
            raise RuntimeError("Socket de segmentación no disponible.")

        # Opción A: configuramos prompts una vez por conexión y los reenviamos tras reconectar.
        payload = encode_configure_request(
            self.semantic_classes.class_names,
            self.semantic_classes.prompt_class_ids,
        )
        self._send_msg(self.seg_sock, payload)

        response = self._recv_msg(self.seg_sock)
        if response is None or len(response) == 0:
            raise RuntimeError("Respuesta vacía al configurar prompts.")

        data = np.load(io.BytesIO(response), allow_pickle=False)
        message_type = decode_message_type(data)
        if message_type == MSG_ERROR:
            raise RuntimeError(str(np.asarray(data["error"]).item()))
        if message_type != MSG_CONFIGURE_ACK:
            raise RuntimeError(f"Respuesta inesperada al configurar prompts: {message_type}")

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
            self._send_msg(self.seg_sock, encode_segment_request(bgr))
            response = self._recv_msg(self.seg_sock)
            if response is None or len(response) == 0:
                raise RuntimeError("respuesta vacia o incompleta")

            data = np.load(io.BytesIO(response), allow_pickle=False)
            message_type = decode_message_type(data)
            if message_type == MSG_ERROR:
                raise RuntimeError(str(np.asarray(data["error"]).item()))
            if message_type != MSG_SEGMENT_RESULT:
                raise RuntimeError(f"Respuesta inesperada del servidor: {message_type}")
            if "class_map" not in data:
                raise RuntimeError("El servidor no devolvio 'class_map'.")

            class_map = np.asarray(data["class_map"], dtype=np.uint8)
            if class_map.ndim != 2:
                raise RuntimeError(f"class_map con shape invalida: {class_map.shape}")

            if class_map.shape[:2] != bgr.shape[:2]:
                class_map = cv2.resize(
                    class_map,
                    (bgr.shape[1], bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            return class_map
        except Exception as exc:
            self.get_logger().warning(f"Fallo de segmentacion bloqueante: {exc}")
            self._close_segmentation_socket()
            time.sleep(self.reconnect_delay_sec)
            return None

    def project_cloud_with_zbuffer(
        self,
        cloud_map_xyz: np.ndarray,
        class_map: np.ndarray,
        camera_info: CameraInfo,
        T_camera_map: np.ndarray,
    ) -> List[ProjectedPoint]:
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        h, w = class_map.shape

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

        visible_points: List[ProjectedPoint] = []
        ys, xs = np.where(index_buffer >= 0)
        for vv, uu in zip(ys, xs):
            i = index_buffer[vv, uu]
            zz = z[i]
            if zz > depth_buffer[vv, uu] + self.z_buffer_margin:
                continue

            px = pts_map[i, 0]
            py = pts_map[i, 1]
            pz = pts_map[i, 2]
            class_id = int(class_map[vv, uu])
            visible_points.append((float(px), float(py), float(pz), class_id))

        return visible_points

    def accumulate_visible_points(self, points: List[ProjectedPoint]) -> None:
        for x, y, z, class_id in points:
            key = self.xyz_to_voxel_key(x, y, z)
            if key not in self.class_votes:
                self.class_votes[key] = {}
            self.class_votes[key][class_id] = self.class_votes[key].get(class_id, 0) + 1

    def publish_accumulated_cloud(self) -> None:
        if self.latest_cloud_points is None or self.latest_cloud_points.shape[0] == 0:
            return

        output_points: List[PointXYZRGBSemantic] = []

        for i in range(self.latest_cloud_points.shape[0]):
            x = float(self.latest_cloud_points[i, 0])
            y = float(self.latest_cloud_points[i, 1])
            z = float(self.latest_cloud_points[i, 2])

            key = self.xyz_to_voxel_key(x, y, z)
            if key in self.class_votes and self.class_votes[key]:
                class_id = self.select_majority_class_id(self.class_votes[key])
            else:
                class_id = self.unknown_class_id

            r, g, b = self.semantic_classes.color_lut[class_id].tolist()
            traversable = int(self.semantic_classes.traversable_lut[class_id])
            traversability = float(self.semantic_classes.traversability_lut[class_id])
            cost = float(self.semantic_classes.cost_lut[class_id])
            rgb_uint32 = self.pack_rgb(int(r), int(g), int(b))
            output_points.append(
                (
                    x,
                    y,
                    z,
                    rgb_uint32,
                    int(class_id),
                    traversable,
                    traversability,
                    cost,
                )
            )

        self.cloud_pub.publish(self.create_cloud_msg(output_points, self.target_frame))

        if self.verbose:
            self.get_logger().info(
                f"Mapa publicado: total_points={len(output_points)}, "
                f"classified_voxels={len(self.class_votes)}"
            )

    def select_majority_class_id(self, votes: Dict[int, int]) -> int:
        return max(votes.items(), key=lambda item: (item[1], -item[0]))[0]

    def xyz_to_voxel_key(self, x: float, y: float, z: float) -> VoxelKey:
        inv = 1.0 / self.color_voxel_size
        return (int(np.floor(x * inv)), int(np.floor(y * inv)), int(np.floor(z * inv)))

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
        node_ids = [int(node.id) for node in msg.nodes]
        return (len(node_ids), min(node_ids), max(node_ids))

    def maybe_reset_state_on_map_reinit(self, msg: MapData) -> None:
        current_sig = self.compute_node_ids_signature(msg)
        previous_sig = self.last_map_node_ids_signature
        self.last_map_node_ids_signature = current_sig

        if previous_sig is None or current_sig is None:
            return

        prev_count, prev_min_id, _ = previous_sig
        curr_count, curr_min_id, curr_max_id = current_sig

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
        self.class_votes.clear()
        self.voxel_attribute_votes.clear()
        self._close_segmentation_socket()
        self.map_dirty = True

    def pose_to_matrix(self, pose: Pose) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.quaternion_to_rotation_matrix(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return matrix

    def transform_to_matrix(self, tf_msg: Transform) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.quaternion_to_rotation_matrix(
            tf_msg.rotation.x,
            tf_msg.rotation.y,
            tf_msg.rotation.z,
            tf_msg.rotation.w,
        )
        matrix[:3, 3] = [tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z]
        return matrix

    def quaternion_to_rotation_matrix(self, x: float, y: float, z: float, w: float) -> np.ndarray:
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

    def pack_rgb(self, r: int, g: int, b: int) -> int:
        return (r << 16) | (g << 8) | b

    def create_cloud_msg(self, points: List[PointXYZRGBSemantic], frame_id: str) -> PointCloud2:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="class_id", offset=16, datatype=PointField.UINT8, count=1),
            PointField(name="traversable", offset=17, datatype=PointField.UINT8, count=1),
            PointField(name="traversability", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="cost", offset=24, datatype=PointField.FLOAT32, count=1),
        ]
        return point_cloud2.create_cloud(header, fields, points)

    def destroy_node(self):
        self._close_segmentation_socket()
        super().destroy_node()


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

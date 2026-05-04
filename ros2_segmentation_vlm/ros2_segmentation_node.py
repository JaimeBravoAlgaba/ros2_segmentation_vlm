#!/usr/bin/env python3
import io
import os
import socket
import struct
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image

from ros2_segmentation_vlm.segmentation_protocol import (
    MSG_CONFIGURE_ACK,
    MSG_ERROR,
    MSG_SEGMENT_RESULT,
    decode_message_type,
    encode_configure_request,
    encode_segment_request,
)
from ros2_segmentation_vlm.semantic_classes import colorize_class_map, load_semantic_classes


def recvall(sock, n: int):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_msg(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    return recvall(sock, msg_len)


def send_msg(sock, data_bytes: bytes):
    msg_len = struct.pack(">I", len(data_bytes))
    sock.sendall(msg_len + data_bytes)


class SegmentationBridgeNode(Node):
    def __init__(self):
        super().__init__("segmentation_bridge_node")

        default_semantic_path = os.path.join(
            get_package_share_directory("ros2_segmentation_vlm"),
            "config",
            "demo_semantic_classes.json",
        )

        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("input_topic", "/camera/color/image_raw")
        self.declare_parameter("output_topic", "/segmentation/color/image")
        self.declare_parameter("class_output_topic", "/segmentation/class")
        self.declare_parameter("reconnect_delay", 1.0)
        self.declare_parameter("qos_depth", 2)
        self.declare_parameter("input_reliability", "best_effort")
        self.declare_parameter("semantic_classes_path", default_semantic_path)

        self.host = self.get_parameter("host").get_parameter_value().string_value
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.class_output_topic = self.get_parameter("class_output_topic").get_parameter_value().string_value
        self.reconnect_delay = self.get_parameter("reconnect_delay").get_parameter_value().double_value
        self.qos_depth = self.get_parameter("qos_depth").get_parameter_value().integer_value
        self.input_reliability = (
            self.get_parameter("input_reliability").get_parameter_value().string_value.strip().lower()
        )
        self.semantic_classes = load_semantic_classes(
            self.get_parameter("semantic_classes_path").get_parameter_value().string_value
        )

        self.bridge = CvBridge()
        self.sock = None

        reliability_policy = QoSReliabilityPolicy.BEST_EFFORT
        if self.input_reliability in ("reliable", "reliability_reliable"):
            reliability_policy = QoSReliabilityPolicy.RELIABLE

        self.input_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=max(1, int(self.qos_depth)),
            reliability=reliability_policy,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._connect_to_server()

        self.image_sub = self.create_subscription(Image, self.input_topic, self.image_callback, self.input_qos)
        self.seg_image_pub = self.create_publisher(Image, self.output_topic, 10)
        self.seg_class_pub = self.create_publisher(Image, self.class_output_topic, 10)

    def _connect_to_server(self):
        try:
            self._close_socket()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            send_msg(
                self.sock,
                encode_configure_request(
                    self.semantic_classes.class_names,
                    self.semantic_classes.prompt_class_ids,
                ),
            )
            response = recv_msg(self.sock)
            if response is None:
                raise RuntimeError("No hubo respuesta al configurar prompts.")
            data = np.load(io.BytesIO(response), allow_pickle=False)
            message_type = decode_message_type(data)
            if message_type == MSG_ERROR:
                raise RuntimeError(str(np.asarray(data["error"]).item()))
            if message_type != MSG_CONFIGURE_ACK:
                raise RuntimeError(f"Respuesta inesperada al configurar prompts: {message_type}")
            self.get_logger().info(f"Connected to segmentation server at {self.host}:{self.port}")
        except Exception as exc:
            self.get_logger().error(f"Failed to connect to segmentation server: {exc}")
            self.sock = None

    def _close_socket(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def image_callback(self, msg: Image):
        if self.sock is None:
            self.get_logger().warn("No connection to segmentation server. Reconnecting.")
            self._connect_to_server()
            time.sleep(self.reconnect_delay)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if not isinstance(cv_image, np.ndarray):
                self.get_logger().error("Input image could not be converted to numpy array.")
                return

            if cv_image.dtype != np.uint8:
                cv_image = cv_image.astype(np.uint8)

            send_msg(self.sock, encode_segment_request(cv_image))
            seg_bytes = recv_msg(self.sock)
            if seg_bytes is None:
                self.get_logger().error("Connection to segmentation server lost.")
                self._close_socket()
                return

            data = np.load(io.BytesIO(seg_bytes), allow_pickle=False)
            message_type = decode_message_type(data)
            if message_type == MSG_ERROR:
                raise RuntimeError(str(np.asarray(data["error"]).item()))
            if message_type != MSG_SEGMENT_RESULT:
                raise RuntimeError(f"Unexpected response type: {message_type}")
            if "class_map" not in data:
                raise RuntimeError("Server response does not contain 'class_map'.")

            class_map = np.asarray(data["class_map"], dtype=np.uint8)
            if class_map.ndim != 2:
                raise RuntimeError(f"Invalid class_map shape from server: {class_map.shape}")

            seg_rgb = colorize_class_map(class_map, self.semantic_classes)
            seg_msg = self.bridge.cv2_to_imgmsg(seg_rgb, encoding="rgb8")
            seg_msg.header = msg.header
            self.seg_image_pub.publish(seg_msg)

            class_msg = self.bridge.cv2_to_imgmsg(class_map, encoding="mono8")
            class_msg.header = msg.header
            self.seg_class_pub.publish(class_msg)

        except CvBridgeError as exc:
            self.get_logger().error(f"CvBridge error: {exc}")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            self.get_logger().error(f"Connection error: {exc}")
            self._close_socket()
        except Exception as exc:
            self.get_logger().error(f"Error during segmentation bridge: {exc}")

    def destroy_node(self):
        self._close_socket()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationBridgeNode()
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

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import socket
import struct
import io
import numpy as np
import time


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

        # Parameters
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("input_topic", "/camera/color/image_raw")
        self.declare_parameter("output_topic", "/segmentation/color/image")
        self.declare_parameter("class_output_topic", "/segmentation/class")
        self.declare_parameter("reconnect_delay", 1.0)
        self.declare_parameter("qos_depth", 2)

        # Options: "best_effort" (recommended), "reliable"
        self.declare_parameter("input_reliability", "best_effort")

        self.host = self.get_parameter("host").get_parameter_value().string_value
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.class_output_topic = self.get_parameter("class_output_topic").get_parameter_value().string_value
        self.reconnect_delay = self.get_parameter("reconnect_delay").get_parameter_value().double_value
        self.qos_depth = self.get_parameter("qos_depth").get_parameter_value().integer_value
        self.input_reliability = (
            self.get_parameter("input_reliability")
            .get_parameter_value()
            .string_value.strip()
            .lower()
        )

        self.get_logger().info(
            f"Segmentation server params: host={self.host}, port={self.port}"
        )

        self.bridge = CvBridge()
        self.sock = None

        # Build QoS for input subscription
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

        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            self.input_qos,
        )

        self.seg_image_pub = self.create_publisher(Image, self.output_topic, 10)
        self.seg_class_pub = self.create_publisher(Image, self.class_output_topic, 10)

        self.get_logger().info(
            f"Segmentation bridge node initialized. "
            f"Subscribing to '{self.input_topic}' with reliability='{self.input_reliability}'. "
            f"Publishing color image to '{self.output_topic}' and class map to '{self.class_output_topic}'."
        )

    def _connect_to_server(self):
        try:
            if self.sock is not None:
                try:
                    self.sock.close()
                except Exception:
                    pass

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.get_logger().info(
                f"Connected to segmentation server at {self.host}:{self.port}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to connect to segmentation server: {e}")
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
            self.get_logger().warn(
                "No connection to segmentation server. "
                "Skipping frame and attempting to reconnect."
            )
            self._connect_to_server()
            time.sleep(self.reconnect_delay)
            return

        try:
            # ROS Image -> OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if not isinstance(cv_image, np.ndarray):
                self.get_logger().error("Input image could not be converted to numpy array.")
                return

            if cv_image.dtype != np.uint8:
                cv_image = cv_image.astype(np.uint8)

            # --- 1) Serialize numpy array with numpy.save (ROS -> server) ---
            buf = io.BytesIO()
            np.save(buf, cv_image, allow_pickle=False)
            send_msg(self.sock, buf.getvalue())

            # --- 2) Receive NPZ response from server ---
            seg_bytes = recv_msg(self.sock)
            if seg_bytes is None:
                self.get_logger().error("Connection to segmentation server lost.")
                self._close_socket()
                return

            # --- 3) Decode NPZ payload ---
            try:
                payload_buf = io.BytesIO(seg_bytes)
                data = np.load(payload_buf, allow_pickle=False)

                if "seg_rgb" not in data or "class_map" not in data:
                    self.get_logger().error(
                        "Server response NPZ does not contain 'seg_rgb' and 'class_map'."
                    )
                    return

                seg_rgb = data["seg_rgb"]
                class_map = data["class_map"]

            except Exception as e:
                self.get_logger().error(f"Failed to decode NPZ response from server: {e}")
                return

            # --- 4) Validate outputs ---
            seg_rgb = np.asarray(seg_rgb)
            class_map = np.asarray(class_map)

            if seg_rgb.dtype != np.uint8:
                seg_rgb = seg_rgb.astype(np.uint8)

            if class_map.dtype != np.uint8:
                class_map = class_map.astype(np.uint8)

            if seg_rgb.ndim != 3 or seg_rgb.shape[2] != 3:
                self.get_logger().error(
                    f"Invalid seg_rgb shape from server: {seg_rgb.shape}, expected (H, W, 3)"
                )
                return

            if class_map.ndim != 2:
                self.get_logger().error(
                    f"Invalid class_map shape from server: {class_map.shape}, expected (H, W)"
                )
                return

            if seg_rgb.shape[:2] != class_map.shape:
                self.get_logger().error(
                    f"Shape mismatch: seg_rgb shape={seg_rgb.shape}, "
                    f"class_map shape={class_map.shape}"
                )
                return

            # --- 5) Publish color segmentation image ---
            seg_msg = self.bridge.cv2_to_imgmsg(seg_rgb, encoding="rgb8")
            seg_msg.header = msg.header
            self.seg_image_pub.publish(seg_msg)

            # --- 6) Publish class index map ---
            class_msg = self.bridge.cv2_to_imgmsg(class_map, encoding="mono8")
            class_msg.header = msg.header
            self.seg_class_pub.publish(class_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            self.get_logger().error(f"Connection error: {e}")
            self._close_socket()
        except Exception as e:
            self.get_logger().error(f"Error during segmentation bridge: {e}")

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
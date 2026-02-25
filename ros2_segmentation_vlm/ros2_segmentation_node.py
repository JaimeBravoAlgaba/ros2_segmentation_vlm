#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import socket
import struct
import io
import numpy as np
import time

HOST = '127.0.0.1'
PORT = 8765  # must match server

def recvall(sock, n):
    data = b''
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
    msg_len = struct.unpack('>I', raw_len)[0]
    return recvall(sock, msg_len)


def send_msg(sock, data_bytes):
    msg_len = struct.pack('>I', len(data_bytes))
    sock.sendall(msg_len + data_bytes)


class SegmentationBridgeNode(Node):
    def __init__(self):
        super().__init__('segmentation_bridge_node')

        # Parameters
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('port', 8765)
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('output_topic', '/segmentation/color/image')
        self.declare_parameter('reconnect_delay', 1.0)
        self.declare_parameter('qos', 10)

        self.host = self.get_parameter('host').get_parameter_value().string_value
        self.port = self.get_parameter('port').get_parameter_value().integer_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.reconnect_delay = self.get_parameter('reconnect_delay').get_parameter_value().double_value
        self.qos = self.get_parameter('qos').get_parameter_value().integer_value

        self.get_logger().info(f"Segmentation server params: host={self.host}, port={self.port}")


        self.bridge = CvBridge()
        self.sock = None

        self._connect_to_server()

        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            self.qos
        )

        self.seg_image_pub = self.create_publisher(
            Image,
            self.output_topic,
            self.qos
        )

        self.get_logger().info('Segmentation bridge node initialized.')

    def _connect_to_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((HOST, PORT))
            self.get_logger().info(f'Connected to segmentation server at {HOST}:{PORT}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to segmentation server: {e}')
            self.sock = None

    def image_callback(self, msg: Image):
        print("Received image for segmentation.")

        if self.sock is None:
            self.get_logger().warn('No connection to segmentation server. Skipping frame and attempting to reconnect.')
            self._connect_to_server()
            time.sleep(self.reconnect_delay)
            return

        try:
            # ROS Image -> OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # --- 1) Serialize numpy array with numpy.save (ROS -> server) ---
            buf = io.BytesIO()
            np.save(buf, cv_image, allow_pickle=False)
            send_msg(self.sock, buf.getvalue())

            # --- 2) Receive segmented image (header + raw bytes) ---
            seg_bytes = recv_msg(self.sock)
            if seg_bytes is None:
                self.get_logger().error('Connection to segmentation server lost.')
                self.sock.close()
                self.sock = None
                return

            if len(seg_bytes) < 12:
                self.get_logger().error(f"Received too few bytes from server: {len(seg_bytes)}")
                return

            # Parse header: H, W, C
            h, w, c = struct.unpack('>III', seg_bytes[:12])
            img_data = seg_bytes[12:]

            expected_data_len = h * w * c
            if len(img_data) != expected_data_len:
                self.get_logger().error(
                    f"Size mismatch from server: got {len(img_data)} bytes, "
                    f"expected {expected_data_len} for shape ({h},{w},{c})"
                )
                return

            # Rebuild numpy array (RGB uint8)
            seg_rgb = np.frombuffer(img_data, dtype=np.uint8).reshape((h, w, c))

            # --- 3) ROS message (encoding: rgb8), keep header/timestamp ---
            seg_msg = self.bridge.cv2_to_imgmsg(seg_rgb, encoding='rgb8')
            seg_msg.header = msg.header

            self.seg_image_pub.publish(seg_msg)

            print(f"Published segmented image at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
        except (BrokenPipeError, ConnectionResetError) as e:
            self.get_logger().error(f'Connection error: {e}')
            if self.sock:
                self.sock.close()
            self.sock = None
        except Exception as e:
            self.get_logger().error(f'Error during segmentation bridge: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.sock:
            node.sock.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
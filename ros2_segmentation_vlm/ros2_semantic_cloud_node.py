#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import message_filters
import image_geometry
import numpy as np
import struct

from sensor_msgs_py import point_cloud2 as pc2


class SemanticCloudNode(Node):
    def __init__(self):
        super().__init__('semantic_cloud_node')

        self.bridge = CvBridge()
        self.cam_model = image_geometry.PinholeCameraModel()

        # Parameters
        self.declare_parameter('input_rgb_topic', '/segmentation/color/image')
        self.declare_parameter('input_depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('output_cloud_topic', '/segmentation/color/points')
        self.declare_parameter('depth_is_16UC1_in_mm', True)  # change if needed
        self.declare_parameter('decimation', 4)               # skip pixels for speed
        self.declare_parameter('queue_size', 5)
        self.declare_parameter('time_slop', 10)               # seconds for ApproximateTimeSynchronizer
    
        seg_topic = self.get_parameter('input_rgb_topic').value
        depth_topic = self.get_parameter('input_depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.output_topic = self.get_parameter('output_cloud_topic').value
        self.depth_is_16UC1_in_mm = self.get_parameter('depth_is_16UC1_in_mm').value
        self.decimation = self.get_parameter('decimation').value
        queue_size = self.get_parameter('queue_size').value
        time_slop = self.get_parameter('time_slop').value

        # Subscribers (message_filters for sync)
        seg_sub = message_filters.Subscriber(self, Image, seg_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [seg_sub, depth_sub, info_sub],
            queue_size=queue_size,
            slop=time_slop # seconds
        )
        self.sync.registerCallback(self.callback)

        self.cloud_pub = self.create_publisher(PointCloud2, self.output_topic, 1)

        self.get_logger().info(f"SemanticCloudNode listening on:\n"
                               f"  seg:   {seg_topic}\n"
                               f"  depth: {depth_topic}\n"
                               f"  info:  {info_topic}\n"
                               f"Publishing: {self.output_topic}")

    def callback(self, seg_msg, depth_msg, info_msg):
        # Update camera model
        self.cam_model.fromCameraInfo(info_msg)

        # Convert images
        seg = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding='rgb8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg)  # encoding depends on camera

        # Normalize depth to float meters
        if self.depth_is_16UC1_in_mm:
            depth = depth.astype(np.float32) / 1000.0  # mm -> m
        else:
            depth = depth.astype(np.float32)           # assume already meters

        h, w = depth.shape
        fx = self.cam_model.fx()
        fy = self.cam_model.fy()
        cx = self.cam_model.cx()
        cy = self.cam_model.cy()

        points = []
        step = self.decimation

        for v in range(0, h, step):
            for u in range(0, w, step):
                z = depth[v, u]
                if z <= 0.0 or np.isinf(z) or np.isnan(z):
                    continue

                x = (u - cx) / fx * z
                y = (v - cy) / fy * z

                # Color from segmentation image
                r, g, b = seg[v, u]
                rgb_uint32 = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]

                points.append((x, y, z, rgb_uint32))

        if not points:
            return

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            ]


        cloud = pc2.create_cloud(
            header=seg_msg.header,  # frame_id = camera frame; RViz can TF it to map
            fields=fields,
            points=points
        )

        self.cloud_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

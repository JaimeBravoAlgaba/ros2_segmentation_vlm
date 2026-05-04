from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    ros_python_env = {"PYTHONNOUSERSITE": "1"}
    pkg_share = get_package_share_directory("ros2_segmentation_vlm")
    default_semantic_classes = os.path.join(
        pkg_share,
        "config",
        "arena_semantic_classes.json",
    )

    segmentation_server_host = LaunchConfiguration("segmentation_server_host")
    segmentation_server_port = LaunchConfiguration("segmentation_server_port")
    semantic_classes_path = LaunchConfiguration("semantic_classes_path")
    map_data_topic = LaunchConfiguration("map_data_topic")
    cloud_map_topic = LaunchConfiguration("cloud_map_topic")
    semantics_cloud_topic = LaunchConfiguration("semantics_cloud_topic")
    traversability_map_topic = LaunchConfiguration("traversability_map_topic")
    traversability_resolution = LaunchConfiguration("traversability_resolution")

    semantics_node = Node(
        package="ros2_segmentation_vlm",
        executable="ros2_semantics_node",
        name="ros2_semantics_node",
        output="screen",
        additional_env=ros_python_env,
        parameters=[
            {
                "segmentation_server_host": segmentation_server_host,
                "segmentation_server_port": segmentation_server_port,
                "semantic_classes_path": semantic_classes_path,
                "map_data_topic": map_data_topic,
                "cloud_map_topic": cloud_map_topic,
                "output_cloud_topic": semantics_cloud_topic,
            }
        ],
    )

    traversability_node = Node(
        package="ros2_segmentation_vlm",
        executable="ros2_traversability_node",
        name="ros2_traversability_node",
        output="screen",
        additional_env=ros_python_env,
        parameters=[
            {
                "input_cloud_topic": semantics_cloud_topic,
                "output_map_topic": traversability_map_topic,
                "resolution": traversability_resolution,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "segmentation_server_host",
                default_value="127.0.0.1",
                description="Segmentation server host",
            ),
            DeclareLaunchArgument(
                "segmentation_server_port",
                default_value="8765",
                description="Segmentation server port",
            ),
            DeclareLaunchArgument(
                "semantic_classes_path",
                default_value=default_semantic_classes,
                description="Path to the semantic classes JSON file",
            ),
            DeclareLaunchArgument(
                "map_data_topic",
                default_value="/rtabmap/mapData",
                description="RTAB-Map mapData topic",
            ),
            DeclareLaunchArgument(
                "cloud_map_topic",
                default_value="/rtabmap/cloud_map",
                description="RTAB-Map cloud map topic",
            ),
            DeclareLaunchArgument(
                "semantics_cloud_topic",
                default_value="/semantics/cloud",
                description="Semantic point cloud topic",
            ),
            DeclareLaunchArgument(
                "traversability_map_topic",
                default_value="/semantics/map/traversability",
                description="Traversability occupancy grid topic",
            ),
            DeclareLaunchArgument(
                "traversability_resolution",
                default_value="0.10",
                description="OccupancyGrid resolution in meters",
            ),
            semantics_node,
            traversability_node,
        ]
    )

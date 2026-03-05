from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # Launch arguments
    host = LaunchConfiguration('host')
    port = LaunchConfiguration('port')
    input_topic = LaunchConfiguration('input_topic')
    output_topic = LaunchConfiguration('output_topic')
    rviz = LaunchConfiguration('rviz')

    # Get RViz config path
    pkg_share = get_package_share_directory('ros2_segmentation_vlm')
    rviz_config = os.path.join(pkg_share, 'rviz', 'ros2_segmentation_vlm.rviz')

    segmentation_bridge = Node(
        package='ros2_segmentation_vlm',
        executable='ros2_segmentation_node',
        name='segmentation_bridge_node',
        output='screen',
        parameters=[{
            'host': host,
            'port': port,
            'input_topic': input_topic,
            'output_topic': output_topic
        }]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(rviz)
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            'host',
            default_value='127.0.0.1',
            description='Segmentation server host'
        ),

        DeclareLaunchArgument(
            'port',
            default_value='8765',
            description='Segmentation server port'
        ),

        DeclareLaunchArgument(
            'input_topic',
            default_value='/camera/color/image_raw',
            description='Input image topic'
        ),

        DeclareLaunchArgument(
            'output_topic',
            default_value='/segmentation/color/image',
            description='Output segmentation topic'
        ),

        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz'
        ),

        segmentation_bridge,
        rviz_node
    ])

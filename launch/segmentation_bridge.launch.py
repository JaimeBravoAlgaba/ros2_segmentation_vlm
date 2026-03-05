from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    host = LaunchConfiguration('host')
    port = LaunchConfiguration('port')
    input_topic = LaunchConfiguration('input_topic')
    output_topic = LaunchConfiguration('output_topic')

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
            description='Output segmented image topic'
        ),

        Node(
            package='ros2_segmentation_vlm',
            executable='ros2_segmentation_node',  # nombre del executable en setup.py
            name='segmentation_bridge_node',
            output='screen',

            parameters=[{
                'host': host,
                'port': port,
                'input_topic': input_topic,
                'output_topic': output_topic
            }]
        )
    ])

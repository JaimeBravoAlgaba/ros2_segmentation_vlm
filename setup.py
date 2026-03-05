from setuptools import find_packages, setup

package_name = 'ros2_segmentation_vlm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/segmentation_bridge.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/ros2_segmentation_vlm.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jaime',
    maintainer_email='jaime.bravo.algaba@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros2_segmentation_node = ros2_segmentation_vlm.ros2_segmentation_node:main',
        ],
    },
)

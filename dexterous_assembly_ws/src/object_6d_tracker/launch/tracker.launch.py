import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # 获取包的安装目录
    pkg_dir = get_package_share_directory("object_6d_tracker")

    # 默认参数文件路径
    default_config_path = os.path.join(pkg_dir, "config", "tracker_params.yaml")

    # 声明可以通过命令行传入的参数
    config_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config_path,
        description="Path to the ROS2 parameters YAML file",
    )

    python_exec_arg = DeclareLaunchArgument(
        "python_exec",
        default_value="/home/x/anaconda3/envs/fdp/bin/python",
        description="Path to the Python interpreter"
    )

    # 定义 6D 追踪节点
    tracker_node = Node(
        package="object_6d_tracker",
        executable="tracker_node",
        name="object_6d_tracker_node",
        output="screen",
        emulate_tty=True,  # 确保终端日志颜色正常显示
        prefix=[LaunchConfiguration("python_exec"), " -u"],  # 使用指定的 Python 解释器并启用无缓冲输出
        parameters=[LaunchConfiguration("config_file")],
        # 如果相机话题名与代码里不一样可重映射
        # remappings=[
        #     ("/rgb/camera_info", "/zed/zed_node/rgb/color/rect/camera_info"),
        #     ("/rgb/image_raw", "/zed/zed_node/rgb/color/rect/image"),
        #     ("/depth_to_rgb/image_raw", "/zed/zed_node/depth/depth_registered"),
        # ],
    )

    return LaunchDescription([config_arg, python_exec_arg, tracker_node])

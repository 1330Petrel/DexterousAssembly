import os
from glob import glob
from setuptools import setup

package_name = "object_6d_tracker"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # 安装 launch 文件
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        # 安装 config 配置文件
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="6D Object Tracking for Dexterous Assembly",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # 定义 ROS2 节点的可执行命令: tracker_node
            "tracker_node = object_6d_tracker.tracker_node:main"
        ],
    },
)

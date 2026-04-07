# DexterousAssembly: Object 6D Tracker (ROS2)

基于 [FoundationPose](https://github.com/NVlabs/FoundationPose) 和 [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) 开发的面向灵巧装配场景的 6D 位姿跟踪系统：

- 远程 Grounded-SAM-2 用于首帧目标分割。
- 本地 FoundationPose 用于首帧配准与后续高速追踪。
- ROS2 TF 实时发布目标位姿，供机械臂/规划模块消费。

## 安装

### 数据准备

1. Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `FoundationPose/weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

2. (Optional) [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `FoundationPose/demo_data/`

### FoundationPose 依赖

```bash
cd DexterousAssembly
conda create -n fdp python=3.10
conda activate fdp

# Install PyTorch (cu128 + pyt27 for 50XX series GPU)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

> 对 30/40XX 系列 GPU，使用 cu124+pyt25：
>
> ```bash
> pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
> ```

```bash
# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
```

在 `FoundationPose/bundlesdf/mycuda/setup.py` 中修改 `include_dirs` 指向 conda 环境中的 Eigen3 路径：

```python
include_dirs=[
    "/path/to/your/eigen3/under/conda",
    "/usr/include/eigen3",
],
```

安装依赖：

```bash
cd FoundationPose
python -m pip install visdom --no-build-isolation
python -m pip install -r requirements.txt

# nvdiffrast 和 pytorch3d
python -m pip install -no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# Build extensions (5090: 12.0, 4090: 8.9, 3090: 8.6)
TORCH_CUDA_ARCH_LIST=12.0 CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

# Kaolin（根据 PyTorch 版本修改对应的链接）
python -m pip install --no-cache-dir kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu128.html
```

如运行时遇到链接错误 `/lib/libgdal.so.30: undefined symbol: TIFFReadRGBATileExt, version LIBTIFF_4.0`，尝试：

```bash
conda install -c conda-forge "gdal>=3.6,<3.8" "libtiff>=4.5,<4.7" --update-deps
```

### ROS2 包安装

```bash
cd dexterous_assembly_ws
conda activate fdp
source /opt/ros/humble/setup.bash
colcon build --packages-select object_6d_tracker
source install/setup.bash
```

## 运行

### 启动 RGB-D 相机（示例：Azure Kinect）

```bash
cd kinect_ros2
source install/setup.bash
ros2 launch azure_kinect_ros_driver driver.launch.py \
 depth_mode:=NFOV_UNBINNED \
 fps:=30 \
 point_cloud:=false \
 rgb_point_cloud:=false
```

### 启动跟踪节点

1. 修改 `src/object_6d_tracker/launch/tracker.launch.py` 中的 Python 解释器路径 `python_exec_arg` 为 `fdp` 环境中的 Python 路径。

2. 根据需要修改 `src/object_6d_tracker/config/tracker_params.yaml` 中的参数。

3. 启动跟踪节点：

    ```bash
    cd dexterous_assembly_ws
    conda activate fdp
    source /opt/ros/humble/setup.bash
    source install/setup.bash

    # 可选：网络缓存调优（弱网/高吞吐场景）
    sudo sysctl -w net.core.rmem_max=26214400
    sudo sysctl -w net.core.rmem_default=26214400

    ros2 launch object_6d_tracker tracker.launch.py
    ```

4. 使用 `rqt_image_view` 观察调试图像：

    ```bash
    ros2 run rqt_image_view rqt_image_view --ros-args -r /image:=/object_6d_tracker_node/debug_image
    ```

5. 重置跟踪：

    ```bash
    ros2 service call /object_6d_tracker_node/reset std_srvs/srv/Trigger
    ```

    或者运行 `tracker_teleop.py` 节点并在终端输入 `esc`：

    ```bash
    ros2 run object_6d_tracker tracker_teleop
    ```

## 关键参数说明

参数文件：`src/object_6d_tracker/config/tracker_params.yaml`

| 参数                 | 说明                        |
| -------------------- | --------------------------- |
| `prompt_mode`        | 获取 SAM 2 prompt 方式      |
| `prompt`             | 目标文本提示词              |
| `debug_level`        | 调试等级（0~3）             |
| `resize_scale`       | 输入缩放比例，降低显存占用  |
| `fp_root_dir`        | FoundationPose 根目录       |
| `obj_path`           | 目标模型 `.obj` 路径        |
| `est_iterations`     | 首帧配准迭代次数            |
| `track_iterations`   | 跟踪迭代次数                |
| `server`             | 远程推理服务器（user@host） |
| `ssh_port`           | 远程 SSH 端口               |
| `remote_python_exec` | 远程 python 路径            |
| `remote_script_path` | 远程推理脚本路径            |
| `ssh_key_path`       | 本地 SSH 私钥路径           |
| `queue_size`         | 同步队列大小                |
| `slop`               | RGB-D 时间同步容忍（秒）    |
| `camera_frame_id`    | TF 父坐标系                 |
| `object_frame_id`    | TF 子坐标系                 |

## 项目结构

```text
dexterous_assembly_ws/
├── src/object_6d_tracker/
│   ├── launch/tracker.launch.py
│   ├── config/tracker_params.yaml
│   ├── object_6d_tracker/
│   │   ├── tracker_node.py
│   │   ├── tracker_teleop.py
│   │   ├── pose_estimator.py
│   │   ├── remote_sam_cli.py
│   │   └── utils.py
│   └── resource/
└── README.md
```

## ACKNOWLEDGEMENTS

- [FoundationPose](https://github.com/NVlabs/FoundationPose)
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)

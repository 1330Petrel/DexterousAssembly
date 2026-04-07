import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from std_srvs.srv import Trigger

import cv2
from enum import Enum
import numpy as np
import queue
import threading
import time
import torch

from .remote_sam_cli import RemoteSAMCLI
from .pose_estimator import PoseEstimator
from . import utils


# 状态流转: WAIT_FIRST_FRAME -> INITIALIZING -> TRACKING
class State(Enum):
    WAIT_FIRST_FRAME = 0
    INITIALIZING = 1
    TRACKING = 2


class TrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("object_6d_tracker_node")

        # ==========================================================
        # 1. 声明并获取 ROS2 参数
        # ==========================================================
        self.declare_parameters(
            namespace="",
            parameters=[
                ("prompt_mode", "text"),  # 'text', 'interactive'
                ("prompt", "wrench"),
                ("debug_level", 2),
                ("resize_scale", 0.375),  # 显存优化：图像缩放比例
                ("fp_root_dir", "/home/x/DexterousAssembly/FoundationPose"),
                (
                    "obj_path",
                    "/home/x/DexterousAssembly/dexterous_assembly_ws/src/object_6d_tracker/resource/models/sam3d/006_wrench.obj",
                ),
                ("est_iterations", 5),
                ("track_iterations", 2),
                ("server", "a@x.x.x.x"),
                ("ssh_port", 0),
                ("remote_python_exec", "/home/x/miniconda3/envs/sam2/bin/python"),
                (
                    "remote_script_path",
                    "linkerhand/Grounded-SAM-2/single_image_infer.py",
                ),
                ("ssh_key_path", "/home/x/.ssh/id_x"),
                ("queue_size", 2),
                ("slop", 0.05),
                ("camera_frame_id", "camera_color_optical_frame"),
                ("object_frame_id", "target_part"),
            ],
        )

        self.p_mode = self.get_parameter("prompt_mode").value
        prompt = self.get_parameter("prompt").value
        if self.p_mode not in ["text", "interactive"]:
            mode = "interactive" if prompt == "" else "text"
            self.get_logger().warn(
                f"Invalid prompt_mode '{self.p_mode}', defaulting to '{mode}'"
            )
            self.p_mode = mode
        if self.p_mode == "text":
            if prompt == "":
                self.get_logger().warn("Got Empty text prompt, switch to interactive")
                self.p_mode = "interactive"
            else:
                self.p_prompt = prompt

        # 打包 Remote SAM CLI 配置
        sam_cfg = {
            "server": self.get_parameter("server").value,
            "ssh_port": self.get_parameter("ssh_port").value,
            "remote_python_exec": self.get_parameter("remote_python_exec").value,
            "remote_script_path": self.get_parameter("remote_script_path").value,
            "ssh_key_path": self.get_parameter("ssh_key_path").value,
        }
        # 打包 Foundation Pose 配置
        fp_cfg = {
            "fp_root_dir": self.get_parameter("fp_root_dir").value,
            "obj_path": self.get_parameter("obj_path").value,
            "est_iterations": self.get_parameter("est_iterations").value,
            "track_iterations": self.get_parameter("track_iterations").value,
            "debug": self.get_parameter("debug_level").value,
        }

        self.p_debug = True if self.get_parameter("debug_level").value > 0 else False
        self.p_resize_scale = self.get_parameter("resize_scale").value

        self.p_cam_frame = self.get_parameter("camera_frame_id").value
        self.p_obj_frame = self.get_parameter("object_frame_id").value

        # ==========================================================
        # 2. 状态机、并发控制与统计
        # ==========================================================
        self.state = State.WAIT_FIRST_FRAME
        self.init_lock = threading.Lock()
        self.init_epoch = 0  # 会话纪元计数器

        # 协同标志位
        self.mask_ready = False
        self.fp_ready = False

        self.first_frame_data = None
        self.first_mask = None

        # 算法占位符
        self.cached_K = None
        self.sam_cli = RemoteSAMCLI(**sam_cfg)
        self.estimator = None
        threading.Thread(
            target=self.task_init_foundation_pose, args=(fp_cfg,), daemon=True
        ).start()  # 后台 FP 初始化线程: 加载 Foundation Pose 环境

        # FPS 统计相关
        self.frame_count = 0
        self.last_fps_time = time.time()

        # ==========================================================
        # 3. 初始化 ROS2 工具
        # ==========================================================
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        info_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
        self.info_sub = self.create_subscription(
            CameraInfo, "/rgb/camera_info", self.info_callback, info_qos
        )

        # TODO 使用后 FPS 反而下降？
        # image_qos = QoSProfile(
        #     reliability=ReliabilityPolicy.BEST_EFFORT,  # 允许丢帧，不重传
        #     durability=DurabilityPolicy.VOLATILE,
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=5,
        # )

        # 订阅话题并进行时间戳软同步
        self.rgb_sub = message_filters.Subscriber(
            self,
            Image,
            "/rgb/image_raw",  # qos_profile=image_qos
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            "/depth_to_rgb/image_raw",  # qos_profile=image_qos
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.get_parameter("queue_size").value,
            slop=self.get_parameter("slop").value,
        )
        self.ts.registerCallback(self.sync_callback)

        # 重置服务
        self.create_service(Trigger, "~/reset", self.reset_callback)

        if self.p_debug:
            # 异步可视化组件
            self.vis_pub = self.create_publisher(Image, "~/debug_image", 10)
            self.vis_queue = queue.Queue(maxsize=2)

        self.get_logger().info(
            f"Object 6D Tracker Node initialized with prompt: '{self.p_prompt}'"
        )

    def info_callback(self, info_msg: CameraInfo) -> None:
        """获取内参后自毁订阅"""
        if self.cached_K is None:
            self.cached_K = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)

            if self.p_resize_scale != 1.0:
                self.cached_K[0:2, :] *= self.p_resize_scale

                h, w = info_msg.height, info_msg.width
                self.new_size = (
                    int(w * self.p_resize_scale),
                    int(h * self.p_resize_scale),
                )
                self.get_logger().info(
                    f"Original image size: ({w}, {h}), resized image size: {self.new_size}"
                )
            else:
                self.new_size = None

            self.get_logger().info("CameraInfo cached: K = \n" + str(self.cached_K))
            # 拿到内参后立刻注销订阅者
            self.destroy_subscription(self.info_sub)

    def sync_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """
        相机传感器同步回调。进入此处的图像和深度图拥有相近的时间戳
        """
        if self.state == State.INITIALIZING or self.cached_K is None:
            # 初始化尚未完成，直接丢弃新到来的帧防止积压
            return

        try:
            # 图像转换与缩放
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            # 对输入下采样以降低显存占用
            if self.new_size:
                new_size = self.new_size
                cv_rgb = cv2.resize(cv_rgb, new_size, interpolation=cv2.INTER_LINEAR)
                cv_depth_mm = cv2.resize(
                    cv_depth_mm, new_size, interpolation=cv2.INTER_NEAREST
                )

            cv_depth_m = cv_depth_mm.astype(np.float32)
            cv_depth_m *= 0.001  # 转换为米

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
            return

        # 检查图像有效性
        if not utils.check_image_is_valid(cv_rgb) or cv_depth_m.size == 0:
            self.get_logger().warning("Invalid RGB or Depth image received, skipping.")
            return

        # ==================== 状态机分发 ====================
        if self.state == State.WAIT_FIRST_FRAME:
            self.handle_initialization(cv_rgb, cv_depth_m, rgb_msg.header.stamp)

        elif self.state == State.TRACKING:
            self.run_tracking(cv_rgb, cv_depth_m, rgb_msg.header.stamp)

    def handle_initialization(self, rgb: np.ndarray, depth: np.ndarray, stamp) -> None:
        self.get_logger().info("Got the first frame, starting initialization...")
        bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.p_mode == "interactive":
            selector = utils.InteractiveSelector()
            prompt_type, prompt_data = selector.get_selection(bgr_image)

            if prompt_type is None:
                self.get_logger().warning("Selection canceled. Wait for next frame...")
                return
        else:
            prompt_type = "text"
            prompt_data = self.p_prompt

        with self.init_lock:
            self.state = State.INITIALIZING
            self.first_frame_data = (rgb, depth, stamp)
            self.init_epoch += 1
            current_epoch = self.init_epoch

            # 启动 SSH 线程: 请求服务器 Grounded-SAM-2 推理
            threading.Thread(
                target=self.task_ssh_sam2,
                args=(bgr_image, prompt_type, prompt_data, current_epoch),
                daemon=True,
            ).start()

    def task_ssh_sam2(
        self, bgr_image: np.ndarray, prompt_type: str, prompt_data, request_epoch: int
    ) -> None:
        """执行远程 SSH 推理"""
        self.get_logger().info("[SSH] Requesting SAM2 Mask via SSH...")
        try:
            # 通信端存 BGR
            mask = self.sam_cli.get_mask(bgr_image, prompt_type, prompt_data)

            if not np.any(mask):
                # 退回 WAIT_FIRST_FRAME 重新尝试
                with self.init_lock:
                    if (
                        self.state == State.INITIALIZING
                        and self.init_epoch == request_epoch
                    ):
                        self.get_logger().warning("[SSH] Got empty mask. Retrying...")
                        self.state = State.WAIT_FIRST_FRAME
                return

            with self.init_lock:
                if self.state != State.INITIALIZING or self.init_epoch != request_epoch:
                    self.get_logger().warning(
                        "[SSH] Received mask for an outdated initialization epoch, discarded"
                    )
                    return

                self.first_mask = mask
                self.mask_ready = True
                self.check_ready_and_register()

        except Exception as e:
            self.get_logger().error(
                f"[SSH][Error] Failed to get mask, please check network and server configuration: {e}"
            )
            # 退回 WAIT_FIRST_FRAME 重新尝试
            with self.init_lock:
                if (
                    self.state == State.INITIALIZING
                    and self.init_epoch == request_epoch
                ):
                    self.state = State.WAIT_FIRST_FRAME

    def task_init_foundation_pose(self, fp_config: dict) -> None:
        """初始化 Foundation Pose 权重"""
        self.get_logger().info("[Local] Initializing Foundation Pose environment...")
        try:
            self.estimator = PoseEstimator(**fp_config)

            with self.init_lock:
                self.fp_ready = True
                self.check_ready_and_register()

        except Exception as e:
            self.get_logger().error(
                f"[Local][Error] Foundation Pose initialization failed: {e}"
            )
            # 直接让节点退出
            if rclpy.ok():
                rclpy.shutdown()
            self.destroy_node()

    def check_ready_and_register(self) -> None:
        """
        同步屏障：只有当服务器的 Mask 返回，且本地模型加载完后，才执行 Register
        """
        if self.mask_ready and self.fp_ready:
            self.get_logger().info("Running first-frame 6D pose Registration...")
            try:
                if self.p_debug:
                    threading.Thread(target=self.vis_worker, daemon=True).start()

                torch.cuda.empty_cache()
                # 调用核心算法的首帧注册
                rgb, depth, stamp = self.first_frame_data
                pose_matrix = self.estimator.register(
                    K=self.cached_K,
                    rgb=rgb,
                    depth=depth,
                    mask=self.first_mask,
                )

                # 发布初始 TF
                self.publish_tf(pose_matrix, stamp)

                # 异步渲染 debug 图像
                if self.p_debug:
                    self.vis_queue.put_nowait((rgb, pose_matrix))

                # 切入追踪模式
                self.get_logger().info(
                    "\033[92mInitialization complete, TRACKING STARTED\033[0m"
                )
                self.state = State.TRACKING
                self.last_fps_time = time.time()
                self.first_frame_data = None
                self.first_mask = None

            except Exception as e:
                self.get_logger().error(f"Registration failed: {e}")
                # 直接让节点退出
                if rclpy.ok():
                    rclpy.shutdown()
                self.destroy_node()

    def run_tracking(self, rgb: np.ndarray, depth: np.ndarray, stamp) -> None:
        """连续帧时序高频追踪逻辑"""
        try:
            # 调用 Foundation Pose 的时序追踪模块
            pose_matrix = self.estimator.track(rgb=rgb, depth=depth, K=self.cached_K)
            # 发布 TF 树
            self.publish_tf(pose_matrix, stamp)

            # 异步渲染 debug 图像
            if self.p_debug and not self.vis_queue.full():
                self.vis_queue.put_nowait((rgb, pose_matrix))

            # FPS 统计
            self.frame_count += 1
            if self.frame_count >= 30:
                now = time.time()
                fps = self.frame_count / (now - self.last_fps_time)
                self.get_logger().info(f"Tracking FPS: {fps:.2f}")
                self.frame_count = 0
                self.last_fps_time = now

        except Exception as e:
            # 为防止少数帧追踪崩溃导致节点死掉，仅做警告
            self.get_logger().warning(f"Skipped a frame due to tracking error: {e}")

    def publish_tf(self, pose_matrix: np.ndarray, stamp) -> None:
        """
        提取 4x4 矩阵的平移和旋转广播至 ROS2 TF 树
        """
        try:
            t_msg = utils.create_transform_stamped(
                matrix=pose_matrix,
                frame_id=self.p_cam_frame,
                child_frame_id=self.p_obj_frame,
                stamp=stamp,
            )
            self.tf_broadcaster.sendTransform(t_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish TF: {e}")

    def reset_callback(self, request, response) -> Trigger.Response:
        result, message = self.reset_tracker()
        response.success = result
        response.message = message
        return response

    def reset_tracker(self) -> tuple[bool, str]:
        """
        退回初始化阶段，清理状态并准备重新捕获第一帧
        """
        if self.state == State.WAIT_FIRST_FRAME:
            return False, "Tracker is already in WAIT_FIRST_FRAME state"

        with self.init_lock:
            if self.state == State.INITIALIZING:
                self.get_logger().warn(
                    "Initialization canceled, Resetting to WAIT_FIRST_FRAME..."
                )
            else:
                self.get_logger().warn(
                    "Tracking stopped, Resetting to WAIT_FIRST_FRAME..."
                )

            # 1. 状态机切回
            self.state = State.WAIT_FIRST_FRAME

            # 2. 清理内参缓存并重新订阅 CameraInfo
            self.cached_K = None
            if hasattr(self, "info_sub") and self.info_sub is not None:
                self.destroy_subscription(self.info_sub)

            info_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
            self.info_sub = self.create_subscription(
                CameraInfo, "/rgb/camera_info", self.info_callback, info_qos
            )

            # 3. 清理数据与掩码缓存
            self.first_frame_data = None
            self.first_mask = None
            self.mask_ready = False

            # 4. 清理 Debug 队列
            while not self.debug_queue.empty():
                try:
                    self.debug_queue.get_nowait()
                except queue.Empty:
                    break

            return True, "Tracker reseted, waiting for new initialization..."

    def vis_worker(self) -> None:
        """后台渲染线程处理 OpenCV 绘制与消息序列化"""
        self.get_logger().info(
            "\033[92m[Vis]\033[0m Visualization Worker started, run 'rqt_image_view /object_6d_tracker_node/debug_image'"
        )
        while rclpy.ok():
            try:
                rgb, pose = self.vis_queue.get(timeout=1.0)

                # 绘制图像
                vis_img = self.estimator.draw_pose_vis(pose, rgb, self.cached_K)
                # 直接按 RGB 编码发布
                vis_msg = self.bridge.cv2_to_imgmsg(vis_img, "rgb8")
                self.vis_pub.publish(vis_msg)

            except queue.Empty:
                pass
            except Exception as e:
                self.get_logger().error(f"[Vis] Failed to render visualization: {e}")

    def destroy_node(self) -> None:
        if self.estimator:
            self.estimator.shutdown()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Got KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

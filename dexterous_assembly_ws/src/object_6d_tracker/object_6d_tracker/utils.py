import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped
from typing import Optional


def matrix_to_tf_components(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    将 4x4 齐次变换矩阵转换为平移向量和四元数 (ROS 格式: x, y, z, w)

    :param matrix: 4x4 numpy 数组
    :return:
        translation: [x, y, z] numpy 数组
        quaternion: [x, y, z, w] numpy 数组
    """
    if matrix.shape != (4, 4):
        raise ValueError(
            f"Invalid input matrix shape: expected (4, 4), got {matrix.shape}"
        )

    # 提取平移部分
    translation = matrix[0:3, 3]

    # 提取旋转部分并转换为四元数
    r = Rotation.from_matrix(matrix[0:3, 0:3])
    quaternion = r.as_quat(canonical=False)  # 默认返回顺序是 ROS 期望的 [x, y, z, w]

    return translation, quaternion


def create_transform_stamped(
    matrix: np.ndarray, frame_id: str, child_frame_id: str, stamp
) -> TransformStamped:
    """
    将 4x4 矩阵打包成 ROS2 的 TransformStamped 消息以便发布到 /tf 树

    :param matrix: 4x4 齐次变换矩阵
    :param frame_id: 父坐标系名称 (通常是相机光心坐标系，如 'camera_color_optical_frame')
    :param child_frame_id: 子坐标系名称 (目标物体坐标系，如 'target_part')
    :param stamp: ROS2 时间戳 (严格使用图像的 header.stamp 以保证 TF 树的高精度同步)
    :return: geometry_msgs.msg.TransformStamped
    """
    t, q = matrix_to_tf_components(matrix)

    t_msg = TransformStamped()
    t_msg.header.stamp = stamp
    t_msg.header.frame_id = frame_id
    t_msg.child_frame_id = child_frame_id

    # 填充平移
    t_msg.transform.translation.x = float(t[0])
    t_msg.transform.translation.y = float(t[1])
    t_msg.transform.translation.z = float(t[2])

    # 填充旋转
    t_msg.transform.rotation.x = float(q[0])
    t_msg.transform.rotation.y = float(q[1])
    t_msg.transform.rotation.z = float(q[2])
    t_msg.transform.rotation.w = float(q[3])

    return t_msg


def check_image_is_valid(image: np.ndarray) -> bool:
    """
    检查图像是否由于传输错误等原因导致全黑或无效

    :param image: numpy 数组图像
    :return: bool
    """
    if image is None or image.size == 0:
        return False
    # 图像极度黑暗（标准差接近 0）也视为无效
    if np.std(image) < 1.0:
        return False
    return True


def get_6d_pose_arr_from_mat(pose_matrix: np.ndarray) -> np.ndarray:
    """
    将 4x4 位姿矩阵转换为 6 自由度数组 [tx, ty, tz, rx, ry, rz] 供卡尔曼滤波使用
    """
    xyz = pose_matrix[:3, 3]
    rotation_matrix = pose_matrix[:3, :3]

    # 使用 'XYZ' 顺序(Roll, Pitch, Yaw) 输出弧度
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)

    # 拼接返回 [tx, ty, tz, rx, ry, rz]
    return np.r_[xyz, euler_angles]


def get_mat_from_6d_pose_arr(pose_arr: np.ndarray) -> np.ndarray:
    """
    将 KF 输出的 6 自由度数组 [tx, ty, tz, rx, ry, rz] 还原回
    4x4 位姿矩阵供 FoundationPose 的 est.pose_last 接收
    """
    xyz = pose_arr[:3]
    euler_angles = pose_arr[3:6]

    # 从欧拉角恢复旋转矩阵
    rotation_matrix = Rotation.from_euler(
        "xyz", euler_angles, degrees=False
    ).as_matrix()

    # 构建 4x4 矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz

    return transformation_matrix


def extract_depth(
    depth_map: np.ndarray,
    centroid_x: int,
    centroid_y: int,
    object_mask: Optional[np.ndarray] = None,
    window_size: int = 5,
) -> float:
    """
    从深度图中提取指定像素的深度值 Z

    :param depth_map: 对齐的深度图 (H, W), 单位为米 (m)
    :param centroid_x: 2D 追踪器提供的物理质心 X
    :param centroid_y: 2D 追踪器提供的物理质心 Y
    :param object_mask: (可选) 2D 追踪器输出的物体掩码，用于剔除背景深度
    :param window_size: 采样窗口大小
    """
    h, w = depth_map.shape
    half = window_size // 2

    # 计算边界安全的窗口范围
    x_min = max(0, centroid_x - half)
    x_max = min(w, centroid_x + half + 1)
    y_min = max(0, centroid_y - half)
    y_max = min(h, centroid_y + half + 1)

    # 提取 ROI 深度块
    patch = depth_map[y_min:y_max, x_min:x_max]

    # 剔除无效深度
    valid_mask = (patch > 0.01) & (patch < 5.0)  # 范围在 0.01m 到 5m 之间
    if object_mask is not None:
        mask_patch = object_mask[y_min:y_max, x_min:x_max]
        valid_mask &= mask_patch > 0  # 只保留掩码内的有效像素
    valid_depths = patch[valid_mask]

    if len(valid_depths) > 0:
        # 使用中值滤波
        return float(np.median(valid_depths))

    # 窗口内全是死像素，返回 -1 表示采样失败
    return -1.0


def get_xyz_from_image(
    K: np.ndarray,
    depth_map: np.ndarray,
    centroid_x: int,
    centroid_y: int,
    old_tz: float,
    object_mask: Optional[np.ndarray] = None,
) -> tuple[float, float, float]:
    """
    根据 2D 追踪器提供的图像像素点 (x, y)，从真实深度图中读取 tz，
    通过相机内参逆投影出 3D (tx, ty, tz)

    :param K: 3x3 相机内参矩阵
    :param depth_map: 当前帧的深度图 (单位: 米)
    :param centroid_x, centroid_y: 追踪器提取出的物体质心像素坐标
    :param old_tz: 上一帧记忆的深度值 (深度采样失败时使用)
    :param object_mask: Cutie 输出的高精度掩码 (用于局部深度采样)
    :return: 修正后的 (tx, ty, tz)
    """
    # 1. 从深度图中采样 tz
    tz = extract_depth(depth_map, centroid_x, centroid_y, object_mask)
    # 深度采样失败沿用 FoundationPose 上一帧记忆的深度
    if tz <= 0.0:
        print(
            f"\033[93m[PoseEstimator][Warning]\033[0m Depth sampling failed at "
            f"({centroid_x}, {centroid_y}); Using old tz: {old_tz:.3f} m"
        )
        tz = old_tz

    # 2. 逆投影出 tx 和 ty
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    tx = (centroid_x - cx) * tz / fx
    ty = (centroid_y - cy) * tz / fy

    return float(tx), float(ty), float(tz)


class InteractiveSelector:
    """OpenCV 交互式框选/点选工具"""

    def __init__(self) -> None:
        self.window_name = "Select Target"
        self._reset()

    def _reset(self) -> None:
        self.drawing = False
        self.done = False
        self.ix, self.iy = -1, -1
        self.fx_mouse, self.fy_mouse = -1, -1
        self.pending_bbox = None
        self.pending_point = None

    def mouse_callback(self, event, x, y, flags, param) -> None:
        if self.done:
            return  # 已完成选择忽略后续事件

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.fx_mouse, self.fy_mouse = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.fx_mouse, self.fy_mouse = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.done = True
            self.fx_mouse, self.fy_mouse = x, y
            dx = abs(self.fx_mouse - self.ix)
            dy = abs(self.fy_mouse - self.iy)

            # 判断是拖拽框还是单点点击
            if dx > 8 and dy > 8:
                x1, y1 = min(self.ix, self.fx_mouse), min(self.iy, self.fy_mouse)
                x2, y2 = max(self.ix, self.fx_mouse), max(self.iy, self.fy_mouse)
                self.pending_bbox = [x1, y1, x2, y2]
            else:
                self.pending_point = [x, y]

    def get_selection(
        self, image_bgr: np.ndarray
    ) -> tuple[Optional[str], Optional[list]]:
        """
        弹出窗口等待操作
        返回: tuple (selection_type, data)
              selection_type: 'bbox' 或 'point'
              data: [x1,y1,x2,y2] 或 [x, y]
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            display = image_bgr.copy()

            # 实时绘制正在拖拽的框
            if self.drawing and self.ix >= 0:
                cv2.rectangle(
                    display,
                    (self.ix, self.iy),
                    (self.fx_mouse, self.fy_mouse),
                    (255, 200, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"({self.ix}, {self.iy}) -> ({self.fx_mouse}, {self.fy_mouse})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # 绘制当前已完成的框/点
            if self.done:
                if self.pending_bbox:
                    x1, y1, x2, y2 = self.pending_bbox
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display,
                        "[ENTER] Confirm  [R] Redraw  [ESC] Cancel",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                elif self.pending_point:
                    px, py = self.pending_point
                    cv2.circle(display, (px, py), 5, (0, 255, 0), -1)
                    cv2.putText(
                        display,
                        "[ENTER] Confirm  [R] Redraw  [ESC] Cancel",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        display,
                        "Drag BBox or Click Point",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    display,
                    "Drag BBox or Click Point",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(50) & 0xFF

            if key == 13 or key == 10:  # Enter
                if self.pending_bbox:
                    print(f"[UI] BBox selected: {self.pending_bbox}")
                    cv2.destroyWindow(self.window_name)
                    return "bbox", self.pending_bbox
                elif self.pending_point:
                    print(f"[UI] Point selected: {self.pending_point}")
                    cv2.destroyWindow(self.window_name)
                    return "point", self.pending_point
                else:
                    print("[UI] No selection made yet")
                    continue
            elif key == 27:  # ESC
                print("[UI] Selection cancelled")
                cv2.destroyWindow(self.window_name)
                return None, None
            elif key == ord("r") or key == ord("R"):  # R
                print("[UI] Redrawing...")
                self._reset()

import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped


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
    rotation_matrix = matrix[0:3, 0:3]
    r = Rotation.from_matrix(rotation_matrix)
    # scipy 的 as_quat() 默认返回顺序正是 ROS 期望的 [x, y, z, w]
    quaternion = r.as_quat()

    return translation, quaternion


def create_transform_stamped(
    matrix: np.ndarray, frame_id: str, child_frame_id: str, stamp
) -> TransformStamped:
    """
    将 4x4 矩阵打包成 ROS2 的 TransformStamped 消息以便发布到 /tf 树

    :param matrix: 4x4 齐次变换矩阵
    :param frame_id: 父坐标系名称 (通常是相机光心坐标系，如 'camera_color_optical_frame')
    :param child_frame_id: 子坐标系名称 (目标物体坐标系，如 'target_part')
    :param stamp: ROS2 时间戳 (必须严格使用图像的 header.stamp 以保证 TF 树的高精度同步)
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
    # 如果图像极度黑暗（标准差接近0），也视为无效
    if np.std(image) < 1.0:
        return False
    return True

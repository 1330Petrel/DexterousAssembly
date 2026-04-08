import numpy as np
import scipy.linalg


class KalmanFilter6D:
    """
    针对 6D 位姿 (平移 + 旋转) 的卡尔曼滤波器

    状态空间 (12维):
    [tx, ty, tz, rx, ry, rz, v_tx, v_ty, v_tz, v_rx, v_ry, v_rz]

    其中：
      - tx, ty, tz: 3D 平移 (通常单位为米)
      - rx, ry, rz: 3D 旋转的欧拉角表达 (通常单位为弧度)
      - v_xxx: 对应的变化速率 (速度)
    """

    def __init__(self, measurement_noise_scale: float = 0.05) -> None:
        """
        初始化 6D 卡尔曼滤波器

        :param measurement_noise_scale: 测量噪声的缩放系数。值越大越不信任当前测量值（越依赖时序预测/滤波效果越强）
        """
        ndim, dt = 6, 1.0  # 6个位姿维度，时间步长为 1.0 (相对帧率)

        # 1. 状态转移矩阵 State Transition Matrix (12x12)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # 2. 观测矩阵 Observation Matrix
        # 全量 6D 更新观测矩阵 (6x12) - 对应 FoundationPose 输出
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 降维 2D (XY) 更新观测矩阵 (2x12) - 对应 2D 追踪器 (Cutie) 输出
        self._update_mat_xy = np.zeros((2, 2 * ndim))
        self._update_mat_xy[0, 0] = 1.0  # tx
        self._update_mat_xy[1, 1] = 1.0  # ty

        # 3. 噪声权重配置 Noise Weights
        # XY 传感器的噪声通常非常小 (2D 框很准)
        self._std_weight_trans_xy = 1.0 / 40.0

        # 过程噪声参数 (针对预测阶段)
        self._std_weight_trans = 1.0 / 10.0  # 平移不确定性
        self._std_weight_rot = 1.0 / 20.0  # 旋转不确定性
        self._std_weight_vel_trans = 1.0 / 20.0  # 平移速度不确定性
        self._std_weight_vel_rot = 1.0 / 40.0  # 旋转速度不确定性

        self.measurement_noise_scale = measurement_noise_scale

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        用首帧的 6D 测量值初始化滤波器状态

        :param measurement: [6,] 数组 [tx, ty, tz, rx, ry, rz]
        :return: (mean, covariance) 初始均值(12维)和初始协方差矩阵(12x12)
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # 初始化不确定度尺度, 避免为 0 导致矩阵奇异
        scale_xyz = max(np.linalg.norm(measurement[:3]), 1e-5)
        scale_rot = max(np.linalg.norm(measurement[3:]), 1e-5)

        # 首帧的初始误差设置得稍微大一些，允许后续快速收敛
        std = np.array(
            [
                # Position
                0.2 * self._std_weight_trans * scale_xyz,
                0.2 * self._std_weight_trans * scale_xyz,
                0.2 * self._std_weight_trans * scale_xyz,
                0.2 * self._std_weight_rot * scale_rot,
                0.2 * self._std_weight_rot * scale_rot,
                0.2 * self._std_weight_rot * scale_rot,
                # Velocity
                1.0 * self._std_weight_vel_trans * scale_xyz,
                1.0 * self._std_weight_vel_trans * scale_xyz,
                1.0 * self._std_weight_vel_trans * scale_xyz,
                1.0 * self._std_weight_vel_rot * scale_rot,
                1.0 * self._std_weight_vel_rot * scale_rot,
                1.0 * self._std_weight_vel_rot * scale_rot,
            ],
            dtype=np.float64,
        )
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        卡尔曼滤波：预测步骤 (Predict)
        根据上一帧的状态推断当前帧的状态
        """
        # 尺度因子通常与深度 tz (mean[2]) 相关，加入 max 防止越界或崩溃
        scale_xyz = max(mean[2], 1e-5)
        scale_rot = max(mean[5], 1e-5)

        std_pos = [
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
        ]

        std_vel = [
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_rot * scale_rot,
            self._std_weight_vel_rot * scale_rot,
            self._std_weight_vel_rot * scale_rot,
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 状态转移推演
        mean = np.dot(mean, self._motion_mat.T)

        # 累加系统过程方差
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """将 12 维状态空间投影到 6 维测量空间"""
        scale_xyz = max(mean[2], 1e-5)
        scale_rot = max(mean[5], 1e-5)

        std = [
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_trans * scale_xyz,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
            self.measurement_noise_scale * self._std_weight_rot * scale_rot,
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = (
            np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
            + innovation_cov
        )

        return projected_mean, projected_cov

    def project_for_xy(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """将 12 维状态空间投影到 2 维 (tx, ty) 测量空间"""
        scale_xy = max(np.linalg.norm(mean[:2]), 1e-5)

        # XY 传感器的噪声模型 (较小)
        std_xy = np.array(
            [
                self._std_weight_trans_xy * scale_xy,
                self._std_weight_trans_xy * scale_xy,
            ],
            dtype=np.float64,
        )
        innovation_cov = np.diag(np.square(std_xy))

        projected_mean = np.dot(self._update_mat_xy, mean)
        projected_cov = (
            np.linalg.multi_dot(
                (self._update_mat_xy, covariance, self._update_mat_xy.T)
            )
            + innovation_cov
        )

        return projected_mean, projected_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        卡尔曼滤波：更新步骤 (全量 6D 更新)
        使用 FoundationPose 的 6D 输出结果修正滤波器状态
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # 计算卡尔曼增益
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

        # 计算新息 (实际测量值 - 投影预测值)
        innovation = measurement - projected_mean

        # 状态均值与协方差更新
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        return new_mean, new_covariance

    def update_from_xy(
        self, mean: np.ndarray, covariance: np.ndarray, measurement_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        卡尔曼滤波：降维更新步骤 (仅更新 tx, ty)
        使用 2D Tracker 投影转换出的 (tx, ty) 强行修正滤波器状态
        """
        proj_mean, proj_cov = self.project_for_xy(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            proj_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat_xy.T).T,
            check_finite=False,
        ).T

        # 使用 2D 追踪器的 (tx, ty) 新息更新 12维状态
        innovation = measurement_xy - proj_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, proj_cov, kalman_gain.T)
        )

        return new_mean, new_covariance

import os
import sys
import shutil
import imageio
import logging
import numpy as np
import time
import torch
import trimesh
from concurrent.futures import ThreadPoolExecutor

from .utils import (
    get_6d_pose_arr_from_mat,
    get_mat_from_6d_pose_arr,
    get_xyz_from_image,
)
from .kalman_filter_6d import KalmanFilter6D
from .vot_wrapper import CutieWrapper


class PoseEstimator:
    """
    FoundationPose++ 融合 3 大模块：
      1. Foundation Pose (6D 几何与纹理微调)
      2. Cutie (高鲁棒 2D 像素级追踪与质心提取)
      3. Kalman Filter 6D (抗抖动时序滤波)
    """

    def __init__(
        self,
        enable_2d_tracker: bool,
        enable_kf: bool,
        fp_root_dir: str,
        obj_path: str,
        est_iterations: int = 5,
        track_iterations: int = 2,
        kf_noise_scale: float = 0.05,
        cutie_seg_threshold: float = 0.1,
        cutie_erosion_size: int = 5,
        cutie_half_precision: bool = False,
        debug: int = 0,
        debug_dir: str = "./tmp/debug_fp",
    ):
        """
        初始化位姿估计器

        :param enable_2d_tracker: 是否启用 Cutie 2D 追踪器
        :param enable_kf: 是否启用 Kalman Filter 6D 时序滤波
        :param fp_root_dir: Foundation Pose 源码根目录的绝对路径
        :param obj_path: 目标物体的 .obj 模型绝对路径
        :param est_iterations: 首帧配准的姿态优化迭代次数
        :param track_iterations: 连续帧追踪的姿态优化迭代次数
        :param kf_noise_scale: Kalman Filter 6D 的过程噪声缩放因子，数值越大越平滑但滞后
        :param cutie_seg_threshold: Cutie 分割置信度阈值
        :param cutie_erosion_size: Cutie 分割结果的形态学腐蚀核大小
        :param cutie_half_precision: 是否开启 Cutie 的 FP16 半精度推理
        :param debug: 调试等级 (0: 关闭, 1: 基础可视化, 2: 保存变换矩阵, 3: 完整调试)
        :param debug_dir: 调试输出文件保存目录
        """
        self.enable_2d_tracker = enable_2d_tracker
        self.enable_kf = enable_kf
        self.est_iterations = est_iterations
        self.track_iterations = track_iterations
        self.debug = debug
        start_time = time.time()

        if self.debug > 1:
            self.debug_dir = debug_dir
            self.frame_count = 0
            self.executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="debug_io"
            )  # 后台线程池用于异步IO操作

            if os.path.isdir(self.debug_dir):
                shutil.rmtree(self.debug_dir)
            os.makedirs(os.path.join(self.debug_dir, "ob_in_cam"), exist_ok=True)

            if self.debug >= 3:
                os.makedirs(os.path.join(self.debug_dir, "track_vis"), exist_ok=True)

        # ==========================================================
        # 1. 动态注入 Foundation Pose 路径并加载模型
        # ==========================================================
        if fp_root_dir not in sys.path:
            sys.path.insert(0, fp_root_dir)
            print(
                f"[PoseEstimator] Added Foundation Pose root directory to sys.path: {fp_root_dir}"
            )

        # 导入 Foundation Pose 内部模块
        try:
            import nvdiffrast.torch as dr
            from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
            from Utils import set_logging_format, set_seed

            if self.debug > 0:
                from Utils import draw_posed_3d_box, draw_xyz_axis

                self._draw_posed_3d_box = draw_posed_3d_box
                self._draw_xyz_axis = draw_xyz_axis

        except ImportError as e:
            print(
                f"\033[91m[PoseEstimator][ERROR]\033[0m Failed to import Foundation Pose modules. Check fp_root_dir parameter: {fp_root_dir}"
            )
            raise e

        set_logging_format(logging.WARNING)
        set_seed(0)

        print(f"[PoseEstimator] Loading 3D object mesh: {obj_path}...")
        self.mesh = trimesh.load_mesh(obj_path)

        # 预计算包围盒用于可视化调试
        if self.debug > 0:
            to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
            self.inv_to_origin = np.linalg.inv(to_origin)
            self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(
                2, 3
            )

        print("[PoseEstimator] Loading Foundation Pose model weights...")
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()

        print("[PoseEstimator] Loading CUDA rasterization context...")
        self.glctx = dr.RasterizeCudaContext()

        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug_dir=debug_dir,
            debug=self.debug,
        )

        # 2. 按需加载 Cutie 2D 追踪器
        if self.enable_2d_tracker:
            print(
                "[PoseEstimator] Initializing Cutie 2D tracker, seg_threshold: "
                f"{cutie_seg_threshold}, erosion_size: {cutie_erosion_size}, half_precision: {cutie_half_precision}"
            )
            self.tracker_2d = CutieWrapper(
                cutie_seg_threshold, cutie_erosion_size, cutie_half_precision
            )

        # 3. 按需加载 6D 卡尔曼滤波器
        if self.enable_kf:
            print(
                f"[PoseEstimator] Initializing Kalman Filter 6D, noise scale: {kf_noise_scale}"
            )
            self.kf = KalmanFilter6D(measurement_noise_scale=kf_noise_scale)

        cost = time.time() - start_time
        print(
            f"\033[92m[PoseEstimator]\033[0m Initialization complete! Time taken: {cost:.2f} seconds."
        )

    def register(
        self,
        K: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        首帧配准：同步初始化 3 大模块的内部状态机确定物体的初始 6D 位姿

        :param K: 相机内参矩阵 (3, 3)
        :param rgb: RGB 彩色图 (H, W, 3), dtype=uint8 (必须是 RGB 格式，非 BGR)
        :param depth: 深度图 (H, W), dtype=float32, 单位：米 (m)
        :param mask: 目标物体的 2D 掩膜 (H, W), dtype=uint8 或 bool
        :return: 4x4 齐次变换矩阵 (物体在相机坐标系下的位姿)
        """
        # 1. Foundation Pose 首帧注册
        ob_mask = mask.astype(bool)
        pose = self.est.register(
            K=K, rgb=rgb, depth=depth, ob_mask=ob_mask, iteration=self.est_iterations
        )
        print(
            f"\033[92m[PoseEstimator]\033[0m Registration complete. Estimated pose:\n{pose}"
        )

        # 2. Cutie 2D 追踪器初始化
        if self.enable_2d_tracker:
            cutie_mask = mask.astype(np.uint8)
            bbox, centroid = self.tracker_2d.initialize(rgb, cutie_mask)
            print(
                f"\033[92m[PoseEstimator]\033[0m Cutie 2D tracker Initial bbox: {bbox}, centroid: {centroid}"
            )

        # 3. Kalman Filter 初始化
        if self.enable_kf:
            pose_6d_arr = get_6d_pose_arr_from_mat(pose)
            self.kf_mean, self.kf_cov = self.kf.initiate(pose_6d_arr)
            print(
                f"\033[92m[PoseEstimator]\033[0m Kalman Filter Initial state:\nMean: {self.kf_mean}\nCovariance:\n{self.kf_cov}"
            )

        if self.debug > 1:
            # 异步执行 IO 操作
            frame_id = self.frame_count
            self.executor.submit(self._save_pose_matrix, frame_id, pose)

            if self.debug >= 3:
                # 保存可视化图像
                self.executor.submit(self._save_debug_image, frame_id, pose, rgb, K)

                import open3d
                from Utils import depth2xyzmap, toOpen3dCloud

                # 保存从模型坐标系变换到相机坐标系变换后的模型
                m = self.mesh.copy()
                m.apply_transform(pose)
                m.export(f"{self.debug_dir}/model_tf.obj")
                # 保存场景点云
                xyz_map = depth2xyzmap(depth, K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
                open3d.io.write_point_cloud(f"{self.debug_dir}/scene.ply", pcd)

        return pose

    def track(self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        连续帧追踪接口：Cutie -> 物理深度锚定 -> KF 平滑 -> FP 局部优化 -> KF 更新

        :param rgb: RGB 彩色图 (H, W, 3), dtype=uint8
        :param depth: 深度图 (H, W), dtype=float32, 单位：米 (m)
        :param K: 相机内参矩阵 (3, 3)
        :return: 4x4 齐次变换矩阵
        """
        # 2D 锁定
        if self.enable_2d_tracker:
            # Cutie 反推物理质心
            centroid_xy, current_mask = self.tracker_2d.track(rgb)

            # 追踪器跟丢退化为纯 FP，否则利用 2D Tracker 和 KF 修正位姿先验
            if centroid_xy[0] > 0 and centroid_xy[1] > 0:
                # Tensor (1,4,4) 或 (4,4) -> Numpy (4,4)
                pose_last_np = self.est.pose_last.squeeze(0).cpu().numpy()

                # 利用 2D 质心与真实深度图反算最新的平移 (tx, ty, tz)
                tx, ty, tz = get_xyz_from_image(
                    K=K,
                    depth_map=depth,
                    centroid_x=centroid_xy[0],
                    centroid_y=centroid_xy[1],
                    old_tz=pose_last_np[2, 3],
                    object_mask=current_mask,
                )

                if not self.enable_kf:
                    # 没开滤波直接用算出的物理坐标覆盖 FP 的历史状态
                    new_prior_pose_np = pose_last_np.copy()
                    new_prior_pose_np[0, 3] = tx
                    new_prior_pose_np[1, 3] = ty
                    new_prior_pose_np[2, 3] = tz
                    self.est.pose_last = (
                        torch.from_numpy(new_prior_pose_np)
                        .unsqueeze(0)
                        .to(self.est.pose_last.device)
                    )

                else:
                    # 开启滤波时利用 2D 反推的 (tx, ty) 修正 KF 的内部状态
                    pose_6d_last = get_6d_pose_arr_from_mat(self.est.pose_last)

                    # 1. 用上一帧位姿更新 KF
                    self.kf_mean, self.kf_cov = self.kf.update(
                        self.kf_mean, self.kf_cov, pose_6d_last
                    )

                    # 2. 取出 2D 锚定的 tx, ty 做降维纠偏
                    self.kf_mean, self.kf_cov = self.kf.update_from_xy(
                        self.kf_mean, self.kf_cov, np.array([tx, ty], dtype=np.float64)
                    )

                    # 3. 将 KF 平滑后的先验 (前 6 维) 转换回矩阵给 FP
                    smoothed_pose_mat = get_mat_from_6d_pose_arr(self.kf_mean[:6])

                    # 深度 tz 相信深度相机的采样而非 KF 预测
                    smoothed_pose_mat[2, 3] = tz

                    # 4. 先验注入 FP
                    self.est.pose_last = (
                        torch.from_numpy(smoothed_pose_mat)
                        .unsqueeze(0)
                        .to(self.est.pose_last.device)
                    )

        # 在正确位置的先验起点上 FP 需要优化旋转和边缘贴合度
        pose = self.est.track_one(
            rgb=rgb, depth=depth, K=K, iteration=self.track_iterations
        )

        if self.enable_kf:
            # 基于最终位姿让卡尔曼滤波器向前推演一步
            self.kf_mean, self.kf_cov = self.kf.predict(self.kf_mean, self.kf_cov)

        if self.debug > 1:
            self.frame_count += 1
            frame_id = self.frame_count

            # 异步执行 IO 操作
            self.executor.submit(self._save_pose_matrix, frame_id, pose)

            if self.debug >= 3:
                # 保存可视化图像
                self.executor.submit(self._save_debug_image, frame_id, pose, rgb, K)

        return pose

    def draw_pose_vis(
        self, pose: np.ndarray, rgb: np.ndarray, K: np.ndarray
    ) -> np.ndarray:
        center_pose = pose @ self.inv_to_origin
        vis = self._draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=self.bbox)
        vis = self._draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=0.1,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        return vis

    def _save_pose_matrix(self, frame_id: int, pose: np.ndarray) -> None:
        np.savetxt(
            os.path.join(self.debug_dir, "ob_in_cam", f"{frame_id}.txt"),
            pose.reshape(4, 4),
        )

    def _save_debug_image(
        self, frame_id: int, pose: np.ndarray, rgb: np.ndarray, K: np.ndarray
    ) -> None:
        vis = self.draw_pose_vis(pose, rgb.copy(), K)
        imageio.imwrite(
            os.path.join(self.debug_dir, "track_vis", f"{frame_id}.png"), vis
        )

    def shutdown(self) -> None:
        if self.debug > 1 and self.executor:
            self.executor.shutdown(wait=True)  # 等待所有任务完成
            print("[PoseEstimator] Debug thread pool shutdown complete")

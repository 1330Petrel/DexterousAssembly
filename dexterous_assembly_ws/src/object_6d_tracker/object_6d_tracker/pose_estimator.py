import os
import sys
import shutil
import imageio
import logging
import numpy as np
import time
import trimesh
from concurrent.futures import ThreadPoolExecutor


class PoseEstimator:
    """
    Foundation Pose 的封装类

      1. 加载目标物体的 3D 网格模型 (.obj)
      2. 初始化 Foundation Pose 的网络权重和 CUDA 栅格化上下文
      3. 提供首帧配准 (register) 和后续时序追踪 (track) 的接口
    """

    def __init__(
        self,
        fp_root_dir: str,
        obj_path: str,
        est_iterations: int = 5,
        track_iterations: int = 2,
        debug: int = 0,
        debug_dir: str = "./tmp/debug_fp",
    ):
        """
        初始化位姿估计器

        :param fp_root_dir: Foundation Pose 源码根目录的绝对路径
        :param obj_path: 目标物体的 .obj 模型绝对路径
        :param est_iterations: 首帧配准的姿态优化迭代次数
        :param track_iterations: 连续帧追踪的姿态优化迭代次数
        :param debug: 调试等级 (0: 关闭, 1: 基础可视化, 2: 保存变换矩阵, 3: 完整调试)
        :param debug_dir: 调试输出文件保存目录
        """
        self.est_iterations = est_iterations
        self.track_iterations = track_iterations
        self.debug = debug

        if self.debug > 1:
            self.debug_dir = debug_dir
            self.frame_count = 0

            # 后台线程池用于异步IO操作
            self.executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="debug_io"
            )
            if os.path.isdir(self.debug_dir):
                shutil.rmtree(self.debug_dir)
            os.makedirs(os.path.join(self.debug_dir, "ob_in_cam"), exist_ok=True)

            if self.debug >= 3:
                os.makedirs(os.path.join(self.debug_dir, "track_vis"), exist_ok=True)

        # 动态将 Foundation Pose 的目录注入到系统环境变量
        if fp_root_dir not in sys.path:
            sys.path.insert(0, fp_root_dir)
            print(
                f"[PoseEstimator] Added Foundation Pose root directory to sys.path: {fp_root_dir}"
            )

        # 路径挂载完成后导入 Foundation Pose 内部模块
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
                f"\033[91m[PoseEstimator ERROR]\033[0m Failed to import Foundation Pose modules. Check fp_root_dir parameter: {fp_root_dir}"
            )
            raise e

        # ==========================================================
        # 加载模型
        # ==========================================================
        start_time = time.time()
        set_logging_format(logging.WARNING)
        set_seed(0)

        print(f"[PoseEstimator] Loading 3D object mesh: {obj_path}...")
        self.mesh = trimesh.load(obj_path)

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
        首帧配准接口：利用 Grounded-SAM-2 的 Mask 确定物体的初始 6D 位姿

        :param K: 相机内参矩阵 (3, 3)
        :param rgb: RGB 彩色图 (H, W, 3), dtype=uint8 (必须是 RGB 格式，非 BGR)
        :param depth: 深度图 (H, W), dtype=float32, 单位：米 (m)
        :param mask: 目标物体的 2D 掩膜 (H, W), dtype=uint8 或 bool
        :return: 4x4 齐次变换矩阵 (物体在相机坐标系下的位姿)
        """
        # Foundation Pose 内部需要布尔类型的 mask
        ob_mask = mask.astype(bool)

        pose = self.est.register(
            K=K, rgb=rgb, depth=depth, ob_mask=ob_mask, iteration=self.est_iterations
        )
        del self.est_iterations
        print(
            f"\033[92m[PoseEstimator]\033[0m Registration complete. Estimated pose:\n{pose}"
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
        连续帧追踪接口：依靠 Foundation Pose 内部维护的时序状态进行高频追踪

        :param rgb: RGB 彩色图 (H, W, 3), dtype=uint8
        :param depth: 深度图 (H, W), dtype=float32, 单位：米 (m)
        :param K: 相机内参矩阵 (3, 3)
        :return: 4x4 齐次变换矩阵
        """
        pose = self.est.track_one(
            rgb=rgb, depth=depth, K=K, iteration=self.track_iterations
        )

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

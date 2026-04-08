import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor


class Tracker2D:
    """2D 追踪器基类"""

    def initialize(
        self, frame: np.ndarray, init_mask: np.ndarray
    ) -> tuple[list[int], tuple[int, int]]:
        raise NotImplementedError

    def track(self, frame: np.ndarray) -> tuple[list[int], tuple[int, int], np.ndarray]:
        raise NotImplementedError


class CutieWrapper(Tracker2D):
    """
    Cutie: Video Object Segmentation Tracker，输出高精度 Mask
    """

    def __init__(
        self,
        cutie_seg_threshold: float = 0.1,
        erosion_size: int = 5,
        half_precision: bool = False,
    ) -> None:
        """
        初始化 Cutie 封装器

        :param cutie_seg_threshold: 分割置信度阈值
        :param erosion_size: 形态学腐蚀核大小，用于过滤遮挡带来的边缘噪声
        :param half_precision: 是否开启 FP16 半精度推理
        """
        self.cutie_seg_threshold = cutie_seg_threshold
        self.half_precision = half_precision

        # 定义腐蚀核
        self.kernel = np.ones((erosion_size, erosion_size), np.uint8)

        try:
            from cutie.inference.inference_core import InferenceCore
            from cutie.utils.get_default_model import get_default_model
        except ImportError as e:
            print("\033[91m[CutieWrapper][ERROR]\033[0m Failed to import Cutie modules")
            raise e

        # 获取 Cutie 模型并配置最大内部尺寸
        print("[CutieWrapper] Loading Cutie model...")
        self.cutie_model = get_default_model().to("cuda")
        self.cutie_model.eval()
        if self.half_precision:
            self.cutie = self.cutie.half()

        self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
        self.processor.max_internal_size = -1  # 480  # 限制内部处理分辨率以保证帧率

        print(
            f"\033[92m[CutieWrapper]\033[0m Initialization complete, max_internal_size: {self.processor.max_internal_size}"
        )

    def initialize(
        self, init_frame: np.ndarray, init_mask: np.ndarray
    ) -> tuple[list[int], tuple[int, int]]:
        """
        首帧初始化

        :param init_frame: RGB 图像 (H, W, 3), dtype=uint8
        :param init_mask: 目标物体的二维二进制 Mask (H, W)
        :return: (bbox_xywh, center_point_xy)
        """
        with torch.inference_mode(), torch.amp.autocast_mode.autocast(
            "cuda", enabled=self.half_precision
        ):
            # 图像预处理
            frame_tensor = to_tensor(init_frame).cuda().float()
            mask_tensor = torch.from_numpy(init_mask).cuda()

            # 提取物体 ID (忽略背景 0)
            objects = np.unique(init_mask)
            objects = objects[objects != 0].tolist()
            if not objects:
                print(
                    "\033[93m[CutieWrapper][WARN]\033[0m No valid object in initial mask, defaulting to ID 1"
                )
                objects = [1]  # 安全保护

            # 将首帧状态写入 Cutie 工作记忆
            self.processor.clear_memory()
            output_prob = self.processor.step(
                frame_tensor, mask_tensor, objects=objects
            )
            mask_pred = self.processor.output_prob_to_mask(
                output_prob, segment_threshold=self.cutie_seg_threshold
            )
            mask_np = mask_pred.cpu().numpy().astype(np.uint8)

        # 解析 Mask 获取边界框和质心
        bbox, centroid = self._parse_output(mask_np)

        # 清空无用显存
        torch.cuda.empty_cache()
        return bbox, centroid

    def track(self, frame: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
        """
        连续帧追踪

        :param frame: RGB 图像 (H, W, 3)
        :return: (centroid_xy, current_mask)
                 - centroid_xy: 用于之后在深度图上采样 Z 轴的绝对中心像素
                 - current_mask: 提供给调优可视化或点云剔除使用
        """
        with torch.inference_mode(), torch.amp.autocast_mode.autocast(
            "cuda", enabled=self.half_precision
        ):
            frame_tensor = to_tensor(frame).cuda().float()

            # Cutie 利用短期+长期记忆自动分割
            output_prob = self.processor.step(frame_tensor)
            mask_pred = self.processor.output_prob_to_mask(
                output_prob, segment_threshold=self.cutie_seg_threshold
            )
            mask_np = mask_pred.cpu().numpy().astype(np.uint8)

        _, centroid = self._parse_output(mask_np)
        return centroid, mask_np

    def _parse_output(self, mask_np: np.ndarray) -> tuple[list[int], tuple[int, int]]:
        """
        解析掩码，获取 BBox 和 质心
        """
        # 1. 腐蚀操作：消除边缘噪声
        mask_np = cv2.erode(mask_np, self.kernel, iterations=1)

        # 2. 提取包围盒
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)

        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
        else:
            # 物体完全被遮挡或离开视野
            print(
                "\033[93m[CutieWrapper][WARN]\033[0m Object lost: Failed to extract BBOX"
            )
            return [-1, -1, 0, 0], (-1, -1)

        # 3. 使用图像矩寻找目标连通域的重心
        M = cv2.moments(mask_np)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # 退化处理：降级为 BBox 中心
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2

        return bbox, (cx, cy)

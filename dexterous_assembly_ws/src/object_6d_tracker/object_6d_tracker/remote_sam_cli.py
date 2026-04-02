import os
import cv2
import time
import subprocess
import numpy as np
from typing import Optional


class RemoteSAMCLI:
    """
    负责本地 ROS2 节点与远程 GPU 服务器通信的客户端类

      1. 保存本地第一帧图像为临时文件
      2. SCP 传送到远程服务器
      3. SSH 触发远程单张图片的推理脚本
      4. SCP 将预测好的 Mask 拉取回本地
      5. 读入内存供 Foundation Pose 使用
    """

    def __init__(
        self,
        server: str,
        ssh_port: int,
        remote_python_exec: str,
        remote_script_path: str,
        ssh_key_path: Optional[str] = None,
    ) -> None:
        """
        初始化通信客户端
        :param server: 远程服务器的 IP 地址与用户名
        :param remote_python_exec: 服务器上 Conda 环境中 Python 解释器的绝对路径
        :param remote_script_path: 服务器上 single_image_infer.py 的绝对路径
        :param ssh_key_path: 本地私钥路径 (如 ~/.ssh/id_rsa)，为空则使用默认系统配置
        :param ssh_port: SSH 端口号
        """
        self.server = server
        self.ssh_port = str(ssh_port)
        self.remote_python_exec = remote_python_exec
        self.remote_script_path = remote_script_path
        self.ssh_key_path = ssh_key_path

        # 本地与远程临时文件路径
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp", exist_ok=True)
        self.local_tmp_img = "./tmp/local_frame.png"
        self.local_tmp_mask = "./tmp/local_mask.png"
        self.remote_tmp_img = "linkerhand/tmp/remote_frame.png"
        self.remote_tmp_mask = "linkerhand/tmp/remote_mask.png"

    def _build_ssh_cmd(self, command: str) -> list[str]:
        """构建基础的 SSH 命令列表"""
        cmd = ["ssh", "-p", self.ssh_port]
        if self.ssh_key_path:
            cmd.extend(["-i", self.ssh_key_path])
        cmd.extend([self.server, command])
        return cmd

    def _build_scp_cmd(self, src: str, dst: str) -> list[str]:
        """构建基础的 SCP 命令列表"""
        cmd = ["scp", "-P", self.ssh_port]
        if self.ssh_key_path:
            cmd.extend(["-i", self.ssh_key_path])
        cmd.extend([src, dst])
        return cmd

    def get_mask(self, image_np: np.ndarray, prompt: str) -> np.ndarray:
        """
        传入本地图像和 Prompt，返回推断的 Mask

        :param image_np: 本地捕获的 RGB 图像 (H, W, 3), OpenCV 格式 (BGR)
        :param prompt: 目标物体的文本描述，如 'hammer'
        :return: 二值化 Mask 图像 (H, W), dtype=uint8
        """
        start_time = time.time()
        print(f"[RemoteSAMCLI] Starting remote inference for prompt: '{prompt}'")

        # 1. 保存图像到本地临时路径
        cv2.imwrite(self.local_tmp_img, image_np)

        try:
            # 2. 上传图像到服务器
            print(f"[RemoteSAMCLI] [1/3] Uploading image...")
            remote_target = f"{self.server}:{self.remote_tmp_img}"
            scp_up_cmd = self._build_scp_cmd(self.local_tmp_img, remote_target)
            subprocess.run(
                scp_up_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )

            # 3. 在服务器上执行推理脚本
            print(f"[RemoteSAMCLI] [2/3] Running inference...")
            infer_cmd = (
                "export HF_ENDPOINT=https://hf-mirror.com && "
                f"{self.remote_python_exec} {self.remote_script_path} "
                f"--img {self.remote_tmp_img} "
                f"--prompt {prompt} "
                f"--out {self.remote_tmp_mask}"
            )
            ssh_exec_cmd = self._build_ssh_cmd(infer_cmd)
            result = subprocess.run(
                ssh_exec_cmd, check=True, capture_output=True, text=True, timeout=120
            )
            for line in result.stdout.strip().split("\n"):
                if line:
                    print(f"  [Server Output] {line}")

            # 4. 从服务器下载 Mask 到本地
            print(f"[RemoteSAMCLI] [3/3] Downloading mask...")
            remote_src = f"{self.server}:{self.remote_tmp_mask}"
            scp_down_cmd = self._build_scp_cmd(remote_src, self.local_tmp_mask)
            subprocess.run(
                scp_down_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60
            )

            # 5. 读取拉取下来的 Mask 返回
            mask = cv2.imread(self.local_tmp_mask, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(
                    f"Failed to read the mask image from local path: {self.local_tmp_mask}"
                )

            cost = time.time() - start_time
            print(
                f"\033[92m[RemoteSAMCLI]\033[0m Remote request successful, total time: {cost:.2f} seconds"
            )
            return mask

        except subprocess.TimeoutExpired as e:
            print(
                f"[RemoteSAMCLI] Process timed out after 120 seconds!\nCommand: {' '.join(e.cmd)}\nOutput: {e.output}"
            )
            raise
        except subprocess.CalledProcessError as e:
            print(
                f"[RemoteSAMCLI] Process execution failed!\nCommand: {' '.join(e.cmd)}\nError message: {e.stderr}"
            )
            raise

        finally:
            # 清理本地的临时 RGB 文件
            if os.path.exists(self.local_tmp_img):
                os.remove(self.local_tmp_img)

            # 清理远程的临时文件
            cleanup_cmd = f"rm -f {self.remote_tmp_img} {self.remote_tmp_mask}"
            ssh_cleanup_cmd = self._build_ssh_cmd(cleanup_cmd)
            try:
                subprocess.run(ssh_cleanup_cmd)
            except:
                pass

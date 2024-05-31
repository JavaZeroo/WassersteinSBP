import sys
import os

# 获取当前目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(current_dir, "pytorch-ssim")

# 将目标目录添加到 sys.path
sys.path.append(target_path)

import pytorch_ssim
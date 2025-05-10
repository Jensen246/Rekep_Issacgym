import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from transform_utils import *

# xyzw格式的四元数
quat = [-0.9237, 0.3828, 0.0069, 0.0029]

# 旋转矩阵
rot_matrix = quat2mat(quat)

print(rot_matrix)

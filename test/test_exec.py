import sys
import os
import pickle
from isaacgym import gymapi
import torch

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from isaac_gen_data_path import IsaacGymDataGenerator

# 加载动作序列
stage_result_list = pickle.load(open(os.path.join('test', 'stage_result_list.pkl'), 'rb'))

# 初始化并执行
isaac_data_generator = IsaacGymDataGenerator()
isaac_data_generator.exec_path(stage_result_list)

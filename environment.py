import numpy as np
import os
import time
import datetime
import cv2
import json
import transform_utils as T
import imageio
import torch
from torch import tensor
from PIL import Image
from utils import (
    bcolors,
    get_clock_time,
    angle_between_rotmat,
    angle_between_quats,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
    convert_quat,
)
class ReKepRealEnv:
    """模拟真实环境的类，替代OmniGibson环境"""
    def __init__(self, config, verbose=False):
        self.config = config

        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])

        # 初始化数据目录
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化关键点追踪系统
        self.keypoints = []
        self._keypoint_registry = None
        self._keypoint2object = None
        
        # 重置关节角
        self.reset_joint_pos = tensor([0.0, -0.7, 0.0, -2.0, 0.0, 1.5, 0.0])
        self.curr_joint_pos = self.reset_joint_pos
        
        # 从hand_pose.json文件中读取初始末端执行器位姿
        pose_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/sensor/hand&base_pose.json')
        
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
            base_pos = np.array(pose_data["base_footprint_pose"][:3])
            base_quat = np.array(pose_data["base_footprint_pose"][3:])
            hand_pos = np.array(pose_data["hand_pose"][:3])
            hand_quat = np.array(pose_data["hand_pose"][3:])
        
        self.init_base_pose_tuple = (base_pos, base_quat)
        self.curr_base_pose = np.concatenate([base_pos, base_quat])
        self.world2robot_homo = T.pose_inv(T.pose2mat(self.init_base_pose_tuple))
        
        self.init_ee_pose_tuple = (hand_pos, hand_quat)
        self.curr_ee_pose = np.concatenate([hand_pos, hand_quat])
            
        
    def get_cam_obs(self):
        """获取所有相机的观测数据"""
        # TODO: 替换为真实相机的观测
        cam_obs = {}
        rgb = Image.open(os.path.join(self.config['camera']['data_dir'], "color.png"))
        rgb = np.array(rgb)
        points = np.load(os.path.join(self.config['camera']['data_dir'], "points_world.npy"))
        mask = np.load(os.path.join(self.config['camera']['data_dir'], "actor_mask.npy"))
        mask = torch.from_numpy(mask).to(torch.int32)
        cam_obs[0] = {'rgb': rgb, 'points': points, 'seg': mask}
        return cam_obs
    
    def register_keypoints(self, keypoints):
        # TODO: 替换为真实关键点追踪系统
        for idx, keypoint in enumerate(keypoints):
            self.keypoints.append(keypoint)
        
    def get_keypoint_positions(self):
        # TODO: 实时获取真实关键点位置
        return np.array(self.keypoints)

    def get_ee_pose(self):
        # TODO: 替换为真实机械臂末端执行器位姿
        return self.curr_ee_pose
    
    def get_ee_pos(self):
        return self.get_ee_pose()[:3]
    
    def get_ee_quat(self):
        return self.get_ee_pose()[3:]
    
    def get_arm_joint_pos(self):
        # TODO: 替换为真实机械臂关节位置
        return self.curr_joint_pos

    def open_gripper(self):
        pass
        # print("open gripper")
        
    def get_object_by_keypoint(self, keypoint_idx):
        return 1
    
    def is_grasping(self, candidate_obj):
        return 1
        
    
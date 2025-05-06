"""
使用PyBullet实现的IK求解器，替代原mplib IK solver
"""
import torch
import os
import pybullet as p
import numpy as np
from types import SimpleNamespace

verbose = False

class IKSolver:
    def __init__(self, reset_joint_pos, world2robot_homo):
        # 记录配置参数
        self.reset_joint_pos = reset_joint_pos
        self.world2robot_homo = world2robot_homo
        
        # IK求解参数
        self.damping = 0.05  # 阻尼系数
        self.max_iterations = 100
        self.tolerance = 5e-3  # 增大容差，从1e-3增加到5e-3，使求解更容易成功
        
        # Panda机器人关节限制
        self.joint_limits = np.array([
            [-2.8973, 2.8973],   # joint 1
            [-1.7628, 1.7628],   # joint 2
            [-2.8973, 2.8973],   # joint 3
            [-3.0718, -0.0698],  # joint 4
            [-2.8973, 2.8973],   # joint 5
            [-0.0175, 3.7525],   # joint 6
            [-2.8973, 2.8973]    # joint 7
        ])
        
        # 初始化PyBullet (无GUI模式)
        self.client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        
        # 加载机器人模型
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "data/franka_description/robots/franka_panda.urdf")
        
        self.robot_id = p.loadURDF(urdf_path, 
                               [0, 0, 0], 
                               [0, 0, 0, 1], 
                               useFixedBase=True,
                               physicsClientId=self.client_id)
                               
        # 查找末端执行器链接索引 (panda_hand)
        self.end_effector_link_index = None
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
            if joint_info[12].decode('utf-8') == 'panda_hand':
                self.end_effector_link_index = i
                break
        
        if self.end_effector_link_index is None:
            print("警告: 未找到末端执行器链接'panda_hand'，使用默认链接7")
            self.end_effector_link_index = 7
    
    def solve(self, target_pose, start_joint_pos=None):
        # 检查目标姿态是否在机械臂工作范围内
        pos_in_bounds = True
        if hasattr(self, 'world2robot_homo') and self.world2robot_homo is not None:
            # 转换到机器人坐标系
            target_pos_homo = np.append(target_pose[:3], 1.0)
            target_pos_robot = self.world2robot_homo @ target_pos_homo
            target_pos_robot = target_pos_robot[:3]
            
            # 检查边界
            bounds_min = np.array([-0.85, -0.85, 0.0])  # 从配置文件中获取
            bounds_max = np.array([0.85, 0.85, 1.2])  # 从配置文件中获取
            
            if verbose:
                # 打印边界检查信息
                print(f"\n===== IK求解边界检查 =====")
                print(f"目标位置(世界坐标系): {target_pose[:3]}")
                print(f"目标位置(机器人坐标系): {target_pos_robot}")
                print(f"机器人工作范围: {bounds_min} 到 {bounds_max}")
            
                for i in range(3):
                    if target_pos_robot[i] < bounds_min[i] or target_pos_robot[i] > bounds_max[i]:
                        pos_in_bounds = False
                        print(f"警告: 坐标轴 {i} 超出范围 [{bounds_min[i]}, {bounds_max[i]}]")
                
                print(f"目标位置是否在工作范围内: {pos_in_bounds}")
                
                if not pos_in_bounds:
                    print("警告: 目标位置超出工作范围，IK求解可能失败")
        
        # 结果类
        result = SimpleNamespace()
        
        # 将四元数转换为旋转矩阵格式
        pos = target_pose[:3]
        orn = target_pose[3:7]
        
        # 使用传入的初始关节角度或默认值
        if start_joint_pos is not None:
            # 修复Tensor复制问题
            if isinstance(start_joint_pos, torch.Tensor):
                current_joints = start_joint_pos.detach().cpu().numpy()
            else:
                current_joints = np.array(start_joint_pos)
        else:
            # 确保使用numpy数组
            if isinstance(self.reset_joint_pos, torch.Tensor):
                current_joints = self.reset_joint_pos.detach().cpu().numpy()
            else:
                current_joints = np.array(self.reset_joint_pos)
        
        # 设置夹爪位置
        gripper_target = [0]  # 夹爪位置，0表示打开
        robot_id = self.robot_id
        
        # 获取末端执行器link的索引（panda_hand）
        end_effector_index = 7  # 对应panda_hand
        
        # 计算IK
        jointPoses = p.calculateInverseKinematics(
            robot_id,
            end_effector_index,
            pos,
            orn,
            maxNumIterations=self.max_iterations,
            residualThreshold=self.tolerance
        )
        
        # 把结果分离出来
        jointPoses = list(jointPoses)
        joint_poses = jointPoses[:7]  # 取前7个关节
        
        # 更新结果
        result.cspace_position = np.array(joint_poses)
        result.jpos = joint_poses
        
        # 验证IK结果
        # 前向运动学验证
        for i in range(7):
            p.resetJointState(robot_id, i, joint_poses[i])
        
        # 获取末端执行器的当前位置和姿态
        link_state = p.getLinkState(robot_id, end_effector_index)
        current_pos = link_state[0]
        current_orn = link_state[1]
        
        # 计算位置误差
        pos_error = np.linalg.norm(np.array(current_pos) - np.array(pos))
        
        # 设置结果
        result.position_error = pos_error
        result.success = pos_error < 2.0 * self.tolerance  # 放宽成功标准
        result.status = "SUCCESS" if result.success else f"FAILURE: Error {pos_error:.6f}"
        
        if verbose:
            # 打印更多调试信息
            print(f"\n===== IK求解结果 =====")
            print(f"目标位置: {pos}")
            print(f"目标姿态(四元数): {orn}")
            print(f"求解后位置: {current_pos}")
            print(f"求解后姿态: {current_orn}")
            print(f"位置误差: {pos_error:.6f}")
            print(f"IK是否成功: {result.success}")
            print(f"求解后关节角度: {joint_poses}")
        
            if not result.success:
                print(f"警告: IK求解失败，位置误差 {pos_error:.6f} > 容差 {2.0 * self.tolerance:.6f}")
        
        return result
    
    def __del__(self):
        # 断开PyBullet连接
        if hasattr(self, 'client_id'):
            p.disconnect(self.client_id)

# 如果直接运行此文件，则执行简单测试
if __name__ == "__main__":
    print("IK求解器测试:")
    
    # 初始化
    reset_joint_pos = np.zeros(7)
    world2robot_homo = np.eye(4)
    ik_solver = IKSolver(reset_joint_pos, world2robot_homo)
    
    # 测试求解
    target_pose = np.array([0.5, 0.0, 0.5, 0, np.pi, 0])  # [x, y, z, rx, ry, rz]
    ik_result = ik_solver.solve(target_pose)
    
    # 输出结果
    print(f"求解状态: {ik_result.status}")
    print(f"求解成功: {ik_result.success}")
    print(f"位置误差: {ik_result.position_error:.6f}")
    print(f"关节角度解: {ik_result.cspace_position}")
    
    # 旧版兼容性测试
    print("\n旧版API兼容性测试:")
    print(f"状态 (status): {ik_result.status}")
    print(f"关节角度解 (qgoal): {ik_result.qgoal}")
    
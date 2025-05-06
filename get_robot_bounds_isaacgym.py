"""
使用PyBullet获取Franka Panda机器人末端执行器的坐标范围
通过随机采样关节角度来确定末端执行器的活动范围
"""
import pybullet as p
import numpy as np
import time
import os

# 创建PyBullet环境
client_id = p.connect(p.DIRECT)  # 无GUI模式，只计算数据
p.setGravity(0, 0, -9.8, physicsClientId=client_id)

# 加载Franka Panda机器人URDF
urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "data/franka_description/robots/franka_panda.urdf")
robot_id = p.loadURDF(urdf_path, 
                   [0, 0, 0], 
                   [0, 0, 0, 1], 
                   useFixedBase=True,
                   physicsClientId=client_id)

# 获取关节信息和末端执行器链接索引
num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
active_joints = []
joint_limits = []

print("机器人关节信息:")
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i, physicsClientId=client_id)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    
    # 只考虑可旋转的关节（类型为0的关节为旋转关节）
    if joint_type == p.JOINT_REVOLUTE:
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        print(f"关节名称: {joint_name}, 限制: [{lower_limit}, {upper_limit}]")
        active_joints.append(i)
        joint_limits.append((lower_limit, upper_limit))

# 查找末端执行器链接 (panda_hand)
end_effector_link_index = None
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i, physicsClientId=client_id)
    if joint_info[12].decode('utf-8') == 'panda_hand':
        end_effector_link_index = i
        print(f"找到末端执行器: {joint_info[12].decode('utf-8')}, 索引: {i}")
        break

if end_effector_link_index is None:
    print("警告: 未找到末端执行器链接'panda_hand'，使用默认链接7")
    end_effector_link_index = 7

# 测试所有关节角度为0时的位置(机器人默认姿态)
print("\n测试默认姿态 - 设置所有关节为0...")
for i in active_joints:
    p.resetJointState(robot_id, i, 0, physicsClientId=client_id)

link_state = p.getLinkState(robot_id, end_effector_link_index, computeForwardKinematics=1, physicsClientId=client_id)
ee_pos = link_state[0]  # 位置
ee_orn = link_state[1]  # 方向（四元数）

print(f"末端执行器在默认姿态:")
print(f"- 位置 (xyz): {ee_pos}")
print(f"- 旋转 (四元数 xyzw): {ee_orn}")

# 随机采样获取末端执行器位置范围
num_samples = 5000
ee_positions = []

print(f"开始采样末端执行器位置 (共{num_samples}个样本)...")
start_time = time.time()

for i in range(num_samples):
    # 为每个活动关节生成随机角度
    for j, joint_idx in enumerate(active_joints):
        lower, upper = joint_limits[j]
        random_angle = np.random.uniform(lower, upper)
        p.resetJointState(robot_id, joint_idx, random_angle, physicsClientId=client_id)
    
    # 获取末端执行器位置
    link_state = p.getLinkState(robot_id, end_effector_link_index, computeForwardKinematics=1, physicsClientId=client_id)
    ee_pos = link_state[0]
    ee_positions.append(ee_pos)
    
    # 显示进度
    if (i+1) % (num_samples // 10) == 0:
        print(f"已完成 {i+1}/{num_samples} 个样本 ({(i+1)/num_samples*100:.1f}%)")
        print(f"样例位置: {ee_pos}")

# 计算边界
ee_positions = np.array(ee_positions)
bounds_min = np.min(ee_positions, axis=0)
bounds_max = np.max(ee_positions, axis=0)

elapsed_time = time.time() - start_time
print(f"\n采样完成，用时 {elapsed_time:.2f} 秒")
print("\n==== 末端执行器活动范围 ====")
print(f"最小坐标: {bounds_min}")
print(f"最大坐标: {bounds_max}")
print(f"尺寸: {bounds_max - bounds_min}")

# 将结果保存到文件
np.savetxt("robot_bounds_isaacgym.txt", np.vstack((bounds_min, bounds_max)), fmt="%.6f")
np.save("ee_positions_isaacgym.npy", ee_positions)  # 保存所有采样点用于可视化
print("\n结果已保存到 robot_bounds_isaacgym.txt 和 ee_positions_isaacgym.npy")

# 断开PyBullet连接
p.disconnect(client_id) 
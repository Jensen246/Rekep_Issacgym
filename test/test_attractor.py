import numpy as np
from isaacgym import gymapi, gymutil

# 初始化 Gym
gym = gymapi.acquire_gym()

# 创建仿真参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 创建 Viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# 添加地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 加载 Franka Panda 机器人资产
asset_root = "urdf/franka_description/robots"  # 替换为您的资产路径
asset_file = "franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 创建环境
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

# 添加机器人 Actor
pose = gymapi.Transform()
actor_handle = gym.create_actor(env, asset, pose, "Franka", 0, 1)

# 获取末端执行器的刚体句柄
ee_name = "panda_hand"
ee_handle = gym.find_actor_rigid_body_handle(env, actor_handle, ee_name)

# 设置 Attractor 属性
attractor_props = gymapi.AttractorProperties()
attractor_props.stiffness = 1000.0
attractor_props.damping = 100.0
attractor_props.axes = gymapi.AXIS_ALL
attractor_props.target = gymapi.Transform()
attractor_props.target.p = gymapi.Vec3(0.5, 0.0, 0.5)  # 目标位置
# 注意: gymapi.Quat使用wxyz格式，而项目中标准为xyzw格式
attractor_props.target.r = gymapi.Quat(0, 0, 0, 1)     # 目标姿态（单位四元数，wxyz格式）

# 创建 Attractor
gym.create_rigid_body_attractor(env, ee_handle, attractor_props)

# 仿真循环
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

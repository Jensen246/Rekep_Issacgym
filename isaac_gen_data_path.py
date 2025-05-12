"""
IsaacGym版本的数据生成和路径执行脚本
迁移自ReKepWithSapien/sapien_gen_data_path.py
👉 已为所有 actor 设置 segmentationId，确保语义分割不再全 0
"""
import numpy as np
import os
import yaml
import imageio
import math
import time
import json
import matplotlib.pyplot as plt
from scipy import interpolate

from isaacgym import gymapi, gymtorch, gymutil
import torch
from transform_utils import euler2quat, convert_quat
from utils import get_config
from ik_solver import IKSolver  # 导入IK求解器

from utils import get_config
import environment


class IsaacGymDataGenerator:
    def __init__(self):
        # 读取配置文件
        self.urdf_path = get_config(config_path='./configs/config.yaml')['urdf_path']
        self.srdf_path = get_config(config_path='./configs/config.yaml')['srdf_path']

        # ========== 基本初始化 ==========
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.sim_params.substeps = 1
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 6
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.01
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            print("Failed to create simulation")
            quit()

        self.create_ground()
        self.camera_data_path = './data/sensor'
        os.makedirs(self.camera_data_path, exist_ok=True)

        self.load_robot()      # 机器人 segId = 1
        self.add_box()         # box1 segId = 101, box2 segId = 102

        self.create_viewer()
        self.create_camera_sensor()

        # 执行一次模拟以完成初始化
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.get_robot_state()
        
        # 初始化视图器
        self.viewer_created = True
        
        # 初始化IK求解器
        global_config = get_config(config_path='./configs/config.yaml')
        self.env = environment.ReKepRealEnv(global_config['env'])        
        self.ik_solver = IKSolver(reset_joint_pos=self.env.reset_joint_pos,
                                  world2robot_homo=self.env.world2robot_homo)

    # --------------------------------------------------------------------------
    #
    # 基础环境
    # --------------------------------------------------------------------------
    #
    def create_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    # --------------------------------------------------------------------------
    #
    # 机器人 & 物体
    # --------------------------------------------------------------------------
    #
    def load_robot(self):
        self.env_spacing = 1.5
        self.num_envs = 1
        self.envs = []

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        asset_root = "./data"
        asset_file = "franka_description/robots/franka_panda.urdf"
        self.urdf_full_path = os.path.join(asset_root, asset_file)
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True

        print(f"加载机器人：{self.urdf_full_path}")
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        
        self.env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, 1)
            self.envs.append(env)

            # segmentationId=0, 忽略机器人
            self.robot_handle = self.gym.create_actor(
                env, self.robot_asset, pose, "robot", i, 0, segmentationId=0)

            props = self.gym.get_actor_dof_properties(env, self.robot_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            props["stiffness"].fill(700.0)
            props["damping"].fill(200.0)
            self.gym.set_actor_dof_properties(env, self.robot_handle, props)
            self.num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
            frank_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
            self.franka_lower_limits = frank_dof_props["lower"]
            self.franka_upper_limits = frank_dof_props["upper"]
            franka_mids = 0.3 * (self.franka_upper_limits + self.franka_lower_limits)
            self.default_dof_pos = np.zeros(self.num_dofs, dtype=np.float32)
            self.default_dof_pos[:7] = franka_mids[:7]
            self.default_dof_pos[7:] = self.franka_upper_limits[7:]
            self.dof_names = [self.gym.get_asset_dof_name(self.robot_asset, j) for j in range(self.num_dofs)]
            self.finger_joints_indices = [j for j, n in enumerate(self.dof_names)
                                          if n in ('panda_finger_joint1', 'panda_finger_joint2')]
            # 获取左指尖的刚体形状属性
            left_finger_shape_props = self.gym.get_actor_rigid_shape_properties(env, self.robot_handle)
            # 获取右指尖的刚体形状属性
            right_finger_shape_props = self.gym.get_actor_rigid_shape_properties(env, self.robot_handle)
            # 设置左指尖的摩擦系数
            for prop in left_finger_shape_props:
                prop.friction = 6.0  # 设置为较高的摩擦系数
            self.gym.set_actor_rigid_shape_properties(self.envs[0], self.robot_handle, left_finger_shape_props)

            # 设置右指尖的摩擦系数
            for prop in right_finger_shape_props:
                prop.friction = 6.0  # 设置为较高的摩擦系数
            self.gym.set_actor_rigid_shape_properties(self.envs[0], self.robot_handle, right_finger_shape_props)

        # 设置初始关节位置
        self.gym.set_actor_dof_position_targets(
            self.envs[0], self.robot_handle, self.default_dof_pos)
        
        for _ in range(100):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

    def add_box(self):
        box_size_small = 0.02
        box_size_large = 0.10
        box_asset_options = gymapi.AssetOptions()
        box_asset_options.density = 100

        box_asset_small = self.gym.create_box(self.sim, 6*box_size_small,
                                              box_size_small, box_size_small, box_asset_options)
        box_asset_large = self.gym.create_box(self.sim, box_size_large,
                                              box_size_large, box_size_large, box_asset_options)

        box1_pose = gymapi.Transform()
        box1_pose.p = gymapi.Vec3(0.3, -0.3, box_size_small / 2 + 0.001)
        # 使用直接的四元数值设置旋转45度（绕Z轴）
        # 四元数格式[x,y,z,w]，绕Z轴旋转45度 = [0, 0, sin(45°/2), cos(45°/2)]
        box1_pose.r = gymapi.Quat(0.0, 0.0, 0.3826834, 0.9238795)

        box2_pose = gymapi.Transform()
        box2_pose.p = gymapi.Vec3(0.3, 0.3, box_size_large / 2 + 0.001)

        # boxes segmentationId=101 / 102
        self.box1_handle = self.gym.create_actor(
            self.envs[0], box_asset_small, box1_pose, "box1", 0, 0, segmentationId=101)
        self.box2_handle = self.gym.create_actor(
            self.envs[0], box_asset_large, box2_pose, "box2", 0, 0, segmentationId=102)

        self.gym.set_rigid_body_color(self.envs[0], self.box1_handle, 0,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
        self.gym.set_rigid_body_color(self.envs[0], self.box2_handle, 0,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))
        
        # 获取 box1 的形状属性
        box1_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[0], self.box1_handle)
        for prop in box1_shape_props:
            prop.friction = 5  # 设置摩擦系数
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.box1_handle, box1_shape_props)

        # 获取 box2 的形状属性
        box2_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[0], self.box2_handle)
        for prop in box2_shape_props:
            prop.friction = 5  # 设置摩擦系数
        self.gym.set_actor_rigid_shape_properties(self.envs[0], self.box2_handle, box2_shape_props)

    # --------------------------------------------------------------------------
    #
    # Viewer & Camera
    # --------------------------------------------------------------------------
    #
    def create_viewer(self):
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 640
        self.camera_props.height = 480
        self.viewer = self.gym.create_viewer(self.sim, self.camera_props)
        if self.viewer is None:
            print("无法创建视图器")
            quit()

        cam_pos = gymapi.Vec3(0.9, 0.0, 0.9)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "PAUSE")

    def create_camera_sensor(self):
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 640
        self.camera_props.height = 480
        self.camera_props.enable_tensors = True
        self.camera_handle = self.gym.create_camera_sensor(self.envs[0], self.camera_props)

        self.cam_pos = gymapi.Vec3(0.35, 0.0, 0.5)
        self.cam_target = gymapi.Vec3(0.3, 0.0, 0.0)
        self.gym.set_camera_location(self.camera_handle, self.envs[0], self.cam_pos, self.cam_target)

    # --------------------------------------------------------------------------
    #
    # 关节&控制
    # --------------------------------------------------------------------------
    #
    def get_robot_state(self):
        self.dof_states = self.gym.get_actor_dof_states(self.envs[0], self.robot_handle, gymapi.STATE_ALL)
        self.dof_positions = [self.dof_states[i]['pos'] for i in range(self.num_dofs)]
        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handle, "panda_hand")
        self.base_footprint = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handle, "panda_link0")
    # --------------------------------------------------------------------------
    #
    # 相机数据
    # --------------------------------------------------------------------------
    #
    def _calculate_intrinsics(self) -> np.ndarray:  # Shape=(3,3)
        horizontal_fov = self.camera_props.horizontal_fov * np.pi / 180
        vertical_fov = 2 * np.arctan(self.camera_props.height / self.camera_props.width * np.tan(horizontal_fov / 2))
        
        f_x = (self.camera_props.width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (self.camera_props.height / 2.0) / np.tan(vertical_fov / 2.0)

        K = np.array(
            [
                [f_x, 0.0, self.camera_props.width / 2.0],
                [0.0, f_y, self.camera_props.height / 2.0],
                [0.0, 0.0, 1.0]
            ],
            dtype=np.float32
        )

        return K

    def _calculate_extrinsics(self) -> np.ndarray:  # Shape=(3,4)
        # 将gymapi.Vec3转换为numpy数组
        cam_pos_np = np.array([self.cam_pos.x, self.cam_pos.y, self.cam_pos.z])
        cam_target_np = np.array([self.cam_target.x, self.cam_target.y, self.cam_target.z])

        # IsaacGym camera coordinate system is x-forward, y-right, z-up
        x_axis = cam_target_np - cam_pos_np
        x_axis = x_axis / np.linalg.norm(x_axis)
        tmp_z_axis = np.array([0, 0, 1])
        y_axis = np.cross(tmp_z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # OpenCV camera coordinate system is x-right, y-down, z-forward
        x_axis_opencv = -y_axis
        y_axis_opencv = -z_axis
        z_axis_opencv = x_axis

        pos = cam_pos_np
        rot = np.stack([x_axis_opencv, y_axis_opencv, z_axis_opencv], axis=1)

        T_c2w = np.eye(4, dtype=np.float32)
        T_c2w[0:3, 0:3] = rot
        T_c2w[0:3, 3] = pos

        T_w2c = np.linalg.inv(T_c2w)[:3, :]
        return T_w2c

    def get_masked_point_cloud(
        self,
        depth: torch.Tensor,
        mask: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        cam_extrinsics: torch.Tensor
    ) -> torch.Tensor:
        H, W = depth.shape
        device = depth.device

        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        
        valid = mask.bool()
        x_valid = x[valid].float()
        y_valid = y[valid].float()
        depth_valid = -depth[valid].float()

        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]

        # u = X/Z*fx + cx
        # v = Y/Z*fy + cy
        X = (x_valid - cx) * depth_valid / fx
        Y = (y_valid - cy) * depth_valid / fy
        Z = depth_valid
        pts_cam = torch.stack((X, Y, Z), dim=1)  

        cam_extrinsics_homo = torch.cat([cam_extrinsics, torch.zeros_like(cam_extrinsics[:1, :])], dim=0)
        cam_extrinsics_homo[3, 3] = 1
        cam_extrinsics_homo_inv = torch.inverse(cam_extrinsics_homo)
        R = cam_extrinsics_homo_inv[:3, :3]
        t = cam_extrinsics_homo_inv[:3, 3]
        pts_world = torch.matmul(pts_cam, R.T) + t

        return pts_world

    def _save_point_cloud(self, depth):
        depth = torch.from_numpy(depth)
        mask = torch.ones_like(depth)
        cam_intrinsics = torch.from_numpy(self._calculate_intrinsics())  # 转换为tensor
        cam_extrinsics = torch.from_numpy(self._calculate_extrinsics())  # 转换为tensor
        pts_world = self.get_masked_point_cloud(depth, mask, cam_intrinsics, cam_extrinsics)
        H, W = depth.shape
        np.save(os.path.join(self.camera_data_path, "points_world.npy"), pts_world.cpu().numpy().reshape(H, W, 3))

    def get_camera_data(self):
        os.makedirs(self.camera_data_path, exist_ok=True)

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)
        color_tensor = gymtorch.wrap_tensor(
            self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_COLOR))
        depth_tensor = gymtorch.wrap_tensor(
            self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_DEPTH))
        seg_tensor = gymtorch.wrap_tensor(
            self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_SEGMENTATION))
        self.gym.end_access_image_tensors(self.sim)

        H, W = self.camera_props.height, self.camera_props.width
        color = color_tensor.clone().cpu().numpy().reshape(H, W, 4)
        depth = depth_tensor.clone().cpu().numpy().reshape(H, W)
        seg = seg_tensor.clone().cpu().numpy().reshape(H, W)

        # 打印相机数据信息
        print("\n--------- 相机数据统计 ---------")
        print(f"彩色图像形状: {color.shape}, 类型: {color.dtype}")
        print(f"深度图形状: {depth.shape}, 类型: {depth.dtype}")
        print(f"深度图数值范围: [{depth.min()}, {depth.max()}]")
        print(f"分割图形状: {seg.shape}, 类型: {seg.dtype}")
        print(f"分割图唯一值: {np.unique(seg)}")
        print("--------------------------------\n")

        imageio.imwrite(os.path.join(self.camera_data_path, "color.png"), color[:, :, :3])
        depth_scaled = np.clip((-depth) * 1000, 0, 65535).astype(np.uint16)
        imageio.imwrite(os.path.join(self.camera_data_path, "depth.png"), depth_scaled)
        np.save(os.path.join(self.camera_data_path, "actor_mask.npy"), seg)
        
        # 生成并保存点云
        self._save_point_cloud(depth)

        # 获取并保存panda hand的6D姿态
        hand_transform = self.gym.get_rigid_transform(self.envs[0], self.hand_handle)
        hand_pose = [
            float(hand_transform.p.x),
            float(hand_transform.p.y),
            float(hand_transform.p.z),
            float(hand_transform.r.x),
            float(hand_transform.r.y),
            float(hand_transform.r.z),
            float(hand_transform.r.w)
        ]
        
        base_footprint_transform = self.gym.get_rigid_transform(self.envs[0], self.base_footprint)
        base_footprint_pose = [
            float(base_footprint_transform.p.x),
            float(base_footprint_transform.p.y),
            float(base_footprint_transform.p.z),
            float(base_footprint_transform.r.x),
            float(base_footprint_transform.r.y),
            float(base_footprint_transform.r.z),
            float(base_footprint_transform.r.w)
        ]
        
        # 将姿态保存为JSON文件
        json_path = os.path.join(self.camera_data_path, "hand&base_pose.json")
        with open(json_path, 'w') as f:
            json.dump({
                "hand_pose": hand_pose,
                "base_footprint_pose": base_footprint_pose
            }, f, indent=4)

        print(f"Camera&robot_pose data saved to {self.camera_data_path}")

    # --------------------------------------------------------------------------
    #
    # 主循环
    # --------------------------------------------------------------------------
    #
    def run(self):
        captured_camera_data = False
        frame_count = 0
        print("开始运行模拟循环...")

        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            self.dof_states = self.gym.get_actor_dof_states(self.envs[0], self.robot_handle, gymapi.STATE_ALL)
            self.dof_positions = [self.dof_states[i]['pos'] for i in range(self.num_dofs)]

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            frame_count += 1
            if not captured_camera_data and frame_count > 100:
                print("模拟稳定，捕获相机数据...")
                self.get_camera_data()
                captured_camera_data = True

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    return

        print("查看器已关闭，清理资源...")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def exec_path(self, stage_result_list):
        """执行所有阶段的路径"""
        # 只有在未创建视图器时才创建
        if not self.viewer_created:
            self.create_viewer()
            self.viewer_created = True

        # 执行每个阶段
        for stage_result in stage_result_list:
            if stage_result['is_grasp_stage']:
                self.set_gripper(1.0)  # 打开夹爪
                # 动态姿态对齐：使用抓取对象 box1（segmentationId=101）的当前姿态调整末端执行器朝向
                box1_state = self.gym.get_actor_rigid_body_states(self.envs[0], self.box1_handle, gymapi.STATE_ALL)
                box2_state = self.gym.get_actor_rigid_body_states(self.envs[0], self.box2_handle, gymapi.STATE_ALL)
                # 获取 box1 和 box2 的平面位置 (x, y)
                px1 = float(box1_state["pose"]["p"]["x"][0]); py1 = float(box1_state["pose"]["p"]["y"][0])
                px2 = float(box2_state["pose"]["p"]["x"][0]); py2 = float(box2_state["pose"]["p"]["y"][0])
                # 根据末端路径最后一点的位置判断目标对象（box1或box2）
                target_x, target_y = stage_result['path'][-1][0], stage_result['path'][-1][1]
                dist1 = (target_x - px1)**2 + (target_y - py1)**2
                dist2 = (target_x - px2)**2 + (target_y - py2)**2
                if dist1 <= dist2:
                    # 目标为 box1，获取 box1 当前姿态四元数 (x, y, z, w)
                    bx = float(box1_state["pose"]["r"]["x"][0]); by = float(box1_state["pose"]["r"]["y"][0])
                    bz = float(box1_state["pose"]["r"]["z"][0]); bw = float(box1_state["pose"]["r"]["w"][0])
                else:
                    # 目标为 box2，获取 box2 当前姿态四元数
                    bx = float(box2_state["pose"]["r"]["x"][0]); by = float(box2_state["pose"]["r"]["y"][0])
                    bz = float(box2_state["pose"]["r"]["z"][0]); bw = float(box2_state["pose"]["r"]["w"][0])
                # 计算目标末端执行器四元数：goal_q = down_q * box_q_inv，其中 down_q = [1, 0, 0, 0]
                inv_bx, inv_by, inv_bz, inv_bw = -bx, -by, -bz, bw   # box_q 的共轭（逆）
                dx, dy, dz, dw = 1.0, 0.0, 0.0, 0.0                 # 朝下参考四元数 down_q
                goal_w = dw * inv_bw - (dx * inv_bx + dy * inv_by + dz * inv_bz)
                goal_x = dw * inv_bx + inv_bw * dx + (dy * inv_bz - dz * inv_by)
                goal_y = dw * inv_by + inv_bw * dy + (dz * inv_bx - dx * inv_bz)
                goal_z = dw * inv_bz + inv_bw * dz + (dx * inv_by - dy * inv_bx)
                # 归一化 goal_q，避免数值误差
                norm = math.sqrt(goal_w**2 + goal_x**2 + goal_y**2 + goal_z**2)
                if norm > 1e-6:
                    goal_w, goal_x, goal_y, goal_z = goal_w/norm, goal_x/norm, goal_y/norm, goal_z/norm
                # 替换抓取阶段末端路径点的姿态为计算得到的 goal_q
                stage_result['path'][-1][3:7] = [goal_x, goal_y, goal_z, goal_w]
            else:
                self.set_gripper(0.0)  # 关闭夹爪

            # 执行路径
            self.plan_and_execute_path(
                stage_result['path'],
                is_grasp_stage=stage_result['is_grasp_stage'],
                is_release_stage=stage_result['is_release_stage']
            )

            if stage_result['is_release_stage']:
                self.set_gripper(1.0)  # 打开夹爪
            else:
                if stage_result['is_grasp_stage']:
                    grasp_pose = stage_result['path'][-1] + np.array([0, 0, -0.02, 0, 0, 0, 0])
                    self.move_to_pose(grasp_pose, is_grasp_stage=True, is_release_stage=False)
                self.set_gripper(0.0)  # 关闭夹爪

        print("所有阶段执行完成")

        # 保持窗口打开，直到用户关闭
        while not self.gym.query_viewer_has_closed(self.viewer):
            # 更新物理和渲染
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            # 检查退出事件
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    return

    def plan_and_execute_path(self, end_effector_path, is_grasp_stage=False, is_release_stage=False):
        """执行端点轨迹
        Args:
            end_effector_path: 末端执行器路径，每个点是一个[x,y,z,qx,qy,qz,qw]数组
            is_grasp_stage: 是否是抓取阶段
            is_release_stage: 是否是释放阶段
        """
        print(f"执行路径: {len(end_effector_path)}个点")
        
        # 遍历路径中的每个位姿
        for i, pose in enumerate(end_effector_path):
            print(f"执行位姿 {i+1}/{len(end_effector_path)}")
            self.move_to_pose(pose, is_grasp_stage, is_release_stage)
            
            # 等待机器人达到位置
            # time.sleep(0.5)

    def move_to_pose(self, target_pose, is_grasp_stage, is_release_stage):
        """使用IK求解器移动到指定位姿
        Args:
            target_pose: 目标位姿，[x,y,z,qx,qy,qz,qw]格式
        """
        # 使用IK求解器计算关节角度
        ik_result = self.ik_solver.solve(target_pose=target_pose, start_joint_pos=self.dof_positions[:7])

        if ik_result.success:
            # 获取计算出的关节角度
            joint_positions = ik_result.cspace_position
            if is_grasp_stage:
                joint_positions = np.concatenate([joint_positions, [0.04, 0.04]]).astype(np.float32)
            elif is_release_stage:
                joint_positions = np.concatenate([joint_positions, [0.0, 0.0]]).astype(np.float32)
            # 设置关节位置目标
            self.gym.set_actor_dof_position_targets(
                self.envs[0], self.robot_handle, joint_positions)
            for _ in range(100):
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                if self.viewer_created:
                    self.gym.draw_viewer(self.viewer, self.sim, False)
                    self.gym.sync_frame_time(self.sim)

        else:
            print(f"IK求解失败: {ik_result.status}")
        
        # 运行模拟一段时间以让机器人移动到目标
        steps = 100
        for _ in range(steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            
            if self.viewer_created:
                self.gym.draw_viewer(self.viewer, self.sim, False)
                self.gym.sync_frame_time(self.sim)

        # 更新机器人状态
        self.get_robot_state()

        # 获取当前末端执行器位姿
        hand_transform = self.gym.get_rigid_transform(self.envs[0], self.hand_handle)

        # 计算位置误差
        pos_error = np.sqrt(
            (hand_transform.p.x - target_pose[0])**2 +
            (hand_transform.p.y - target_pose[1])**2 +
            (hand_transform.p.z - target_pose[2])**2
        )

        # 计算姿态误差（四元数）
        quat_error = np.sqrt(
            (hand_transform.r.x - target_pose[3])**2 +
            (hand_transform.r.y - target_pose[4])**2 +
            (hand_transform.r.z - target_pose[5])**2 +
            (hand_transform.r.w - target_pose[6])**2
        )

        # 打印位姿对比和误差
        print("\n--------- 位姿对比 ---------")
        print(f"目标位置: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]")
        print(f"实际位置: [{hand_transform.p.x:.4f}, {hand_transform.p.y:.4f}, {hand_transform.p.z:.4f}]")
        print(f"位置误差: {pos_error:.4f}")
        print(f"目标姿态: [{target_pose[3]:.4f}, {target_pose[4]:.4f}, {target_pose[5]:.4f}, {target_pose[6]:.4f}]")
        print(f"实际姿态: [{hand_transform.r.x:.4f}, {hand_transform.r.y:.4f}, {hand_transform.r.z:.4f}, {hand_transform.r.w:.4f}]")
        print(f"姿态误差: {quat_error:.4f}")
        print("---------------------------\n")

    def set_gripper(self, position, steps=50):
        """设置夹爪位置
        
        Args:
            position: 位置值，0表示闭合，1表示打开
            steps: 执行步数
        """
        current_positions = self.dof_positions[-2:]
        target_positions = [position, position]
        
        # 创建完整的关节位置数组
        full_targets = np.array(self.dof_positions)
        
        for i in range(steps):
            t = (i + 1) / steps
             
            # 线性插值
            interp_positions = [
                current_positions[j] * (1-t) + target_positions[j] * t
                for j in range(2)
            ]
            
            # 更新夹爪关节位置
            for j, idx in enumerate(self.finger_joints_indices):
                full_targets[idx] = interp_positions[j]
            
            # 设置所有关节位置目标
            self.gym.set_actor_dof_position_targets(self.envs[0], self.robot_handle, full_targets)
            
            # 执行模拟步骤
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            
            # 更新显示
            if self.viewer_created:
                self.gym.draw_viewer(self.viewer, self.sim, False)
                self.gym.sync_frame_time(self.sim)
        
        # 更新当前夹爪位置
        self.get_robot_state()


if __name__ == "__main__":
    isaac_data_generator = IsaacGymDataGenerator()
    print("开始运行模拟...")
    isaac_data_generator.run()
    print("模拟结束")

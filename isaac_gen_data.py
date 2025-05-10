"""
IsaacGym版本的数据生成和路径执行脚本
迁移自RekepWithSapien/sapien_gen_data_path.py
👉 已为所有 actor 设置 segmentationId，确保语义分割不再全 0
"""
import numpy as np
import os
import yaml
import imageio
import math

from isaacgym import gymapi, gymtorch, gymutil
import torch
from transform_utils import euler2quat, convert_quat
from utils import get_config


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

    # -------------------------------------------------------------------------- #
    # 基础环境
    # -------------------------------------------------------------------------- #
    def create_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    # -------------------------------------------------------------------------- #
    # 机器人 & 物体
    # -------------------------------------------------------------------------- #
    def load_robot(self):
        self.env_spacing = 1.5
        self.num_envs = 1
        self.envs = []

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        asset_root = "./data"
        asset_file = "franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True

        print(f"加载机器人：{os.path.join(asset_root, asset_file)}")
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
            props["stiffness"].fill(1000.0)
            props["damping"].fill(200.0)
            self.gym.set_actor_dof_properties(env, self.robot_handle, props)

            self.num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
            self.dof_names = [self.gym.get_asset_dof_name(self.robot_asset, j) for j in range(self.num_dofs)]
            self.finger_joints_indices = [j for j, n in enumerate(self.dof_names)
                                          if n in ('panda_finger_joint1', 'panda_finger_joint2')]

    def add_box(self):
        box_size_small = 0.03
        box_size_large = 0.10
        box_asset_options = gymapi.AssetOptions()
        box_asset_options.density = 100

        box_asset_small = self.gym.create_box(self.sim, box_size_small,
                                              box_size_small, box_size_small, box_asset_options)
        box_asset_large = self.gym.create_box(self.sim, box_size_large,
                                              box_size_large, box_size_large, box_asset_options)

        box1_pose = gymapi.Transform()
        box1_pose.p = gymapi.Vec3(0.5, -0.5, box_size_small / 2 + 0.001)

        box2_pose = gymapi.Transform()
        box2_pose.p = gymapi.Vec3(0.5, 0.5, box_size_large / 2 + 0.001)

        # boxes segmentationId=101 / 102
        self.box1_handle = self.gym.create_actor(
            self.envs[0], box_asset_small, box1_pose, "box1", 0, 0, segmentationId=101)
        self.box2_handle = self.gym.create_actor(
            self.envs[0], box_asset_large, box2_pose, "box2", 0, 0, segmentationId=102)

        self.gym.set_rigid_body_color(self.envs[0], self.box1_handle, 0,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
        self.gym.set_rigid_body_color(self.envs[0], self.box2_handle, 0,
                                      gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

    # -------------------------------------------------------------------------- #
    # Viewer & Camera
    # -------------------------------------------------------------------------- #
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

        self.cam_pos = gymapi.Vec3(0.7, 0.0, 1.2)
        self.cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.set_camera_location(self.camera_handle, self.envs[0], self.cam_pos, self.cam_target)

    # -------------------------------------------------------------------------- #
    # 关节&控制
    # -------------------------------------------------------------------------- #
    def get_robot_state(self):
        self.dof_states = self.gym.get_actor_dof_states(self.envs[0], self.robot_handle, gymapi.STATE_ALL)
        self.dof_positions = [self.dof_states[i]['pos'] for i in range(self.num_dofs)]
        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handle, "panda_hand")

    # -------------------------------------------------------------------------- #
    # 相机数据
    # -------------------------------------------------------------------------- #
    def _calculate_intrinsics(self)->np.ndarray: # Shape=(3,3) 
        horizontal_fov = self.camera_props.horizontal_fov * np.pi / 180
        vertical_fov = 2*np.arctan(self.camera_props.height / self.camera_props.width * np.tan(horizontal_fov / 2))
        
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

    def _calculate_extrinsics(self)->np.ndarray: # Shape=(3,4)
        # 将gymapi.Vec3转换为numpy数组
        cam_pos_np = np.array([self.cam_pos.x, self.cam_pos.y, self.cam_pos.z])
        cam_target_np = np.array([self.cam_target.x, self.cam_target.y, self.cam_target.z])

        # IsaacGym camera coordinate system is x-forward, y-right, z-up
        x_axis = cam_target_np - cam_pos_np
        x_axis = x_axis / np.linalg.norm(x_axis)
        tmp_z_axis = np.array([0,0,1])
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

        # u = X/Z*fx +cx
        # v = Y/Z*fy +cy
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
        H,W = depth.shape
        np.save(os.path.join(self.camera_data_path, "points_world.npy"), 
                pts_world.cpu().numpy().reshape(H,W,3))
    
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

        print(f"Camera data saved to {self.camera_data_path}")

    # -------------------------------------------------------------------------- #
    # 主循环
    # -------------------------------------------------------------------------- #
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


if __name__ == "__main__":
    isaac_data_generator = IsaacGymDataGenerator()
    print("开始运行模拟...")
    isaac_data_generator.run()
    print("模拟结束")
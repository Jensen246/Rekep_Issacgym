import numpy as np
import os,json,argparse,pickle

from isaac_gen_data_path import IsaacGymDataGenerator
import torch
from environment import ReKepRealEnv
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
import transform_utils as T

from ik_solver import IKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver

from visualizer import Visualizer


from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

class Main:
    def __init__(self, visualize=True, use_cached_query=False):
        global_config = get_config(config_path='./configs/config.yaml')
        self.config = global_config['main']
        
        self.visualize = visualize
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'], use_cached_query=use_cached_query)
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        
        self.env = ReKepRealEnv(global_config['env'])
        ik_solver = IKSolver(
            reset_joint_pos=self.env.reset_joint_pos,
            world2robot_homo=self.env.world2robot_homo,
        )
        # initialize solvers
        self.ik_solver = ik_solver
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.env.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.env.reset_joint_pos)
        # initialize visualizer
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'], self.env)
                 
    
    def perform_task(self, instruction, rekep_program_dir=None):
        
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[self.config['vlm_camera']]['rgb']
        points = cam_obs[self.config['vlm_camera']]['points']
        mask = cam_obs[self.config['vlm_camera']]['seg']
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        if rekep_program_dir is None:
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, mask)
            print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
            if self.visualize:
                self.visualizer.show_img(projected_img)
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        # ====================================
        # = execute
        # ====================================
        self._execute(rekep_program_dir)
        
    def _execute(self, rekep_program_dir):
        # load metadata
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        # register keypoints to be tracked
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict
            
        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable
        
        # main loop
        # self.last_sim_step_counter = -np.inf
        self.stage_result_list = []
        self._update_stage(1)
        while True:
            stage_result = dict()
            stage_result['is_grasp_stage'] = self.is_grasp_stage    
            stage_result['is_release_stage'] = self.is_release_stage
            
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)  # first keypoint is always the ee
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_pos()
            
            next_subgoal, ik_result = self._get_next_subgoal(from_scratch=self.first_iter)
            next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
            
            self.first_iter = False
            stage_result['subgoal_pose'] = next_subgoal
            stage_result['path'] = next_path[:, :7]
            stage_result['ik_result'] = ik_result
            
            self.stage_result_list.append(stage_result)
            
            # 完成所有阶段后执行，再退出
            if self.stage == self.program_info['num_stages']:
                # print(self.stage_result_list)
                pickle.dump(self.stage_result_list, open(os.path.join('test', 'stage_result_list.pkl'), 'wb'))
                # self.exec_path(self.stage_result_list)
                return
            self.curr_joint_pos = ik_result.cspace_position
            self.env.curr_joint_pos = self.curr_joint_pos
            self.curr_ee_pose = next_subgoal
            self.env.curr_ee_pose = self.curr_ee_pose
            # 进入下一阶段
            self._update_stage(self.stage + 1)
        
    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1            # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        
        # update keypoint movable mask
        # self._update_keypoint_movable_mask()
        self.first_iter = True

    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            # self.sdf_voxels,
                                                            # self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([0, 0, -self.config['grasp_depth'] / 2.0])
        elif self.is_release_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([0, 0, -self.config['grasp_depth'] / 2.0])
        
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        ik_result = self.ik_solver.solve(subgoal_pose, self.curr_joint_pos)
        return subgoal_pose, ik_result

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        
        # # add gripper action
        # ee_action_seq = np.zeros((dense_path.shape[0], 8))
        # ee_action_seq[:, :7] = dense_path
        # # ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return dense_path
    
    def _get_next_path(self, next_subgoal, from_scratch):
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.keypoint_movable_mask,
                                                    path_constraints,
                                                    # self.sdf_voxels,
                                                    # self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        # processed_path = self._process_path(path)
        # if self.visualize:
        #     self.visualizer.visualize_path(processed_path)
        # return processed_path
        if self.visualize:
            self.visualizer.visualize_path(path)
        return path

    def exec_path(self, stage_result_list):
        """执行规划的路径
        
        Args:
            stage_result_list: 包含各阶段路径信息的列表
        """
        isaac_data_generator = IsaacGymDataGenerator()
        isaac_data_generator.exec_path(stage_result_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cube', help='task to perform')
    parser.add_argument('--use_cached_query', default=True, action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--visualize', default=True, action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    task_list = {
        'cube': {
            'instruction': 'put the red cube on the green cube',
            'rekep_program_dir': './vlm_query/cube',
            },
    }
    task = task_list[args.task]
    instruction = task['instruction']
    main = Main(visualize=args.visualize, use_cached_query=args.use_cached_query)
    main.perform_task(instruction,
                    rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None)
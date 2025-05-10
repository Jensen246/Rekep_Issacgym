"""
使用 PyBullet 实现的 IK 求解器（Panda 7‑DoF），采用雅可比矩阵伪逆迭代法求解。
保持原始接口，去掉多余逻辑与错误处理。
"""
import os, math
from types import SimpleNamespace
import numpy as np
import pybullet as p

# Panda joint limits (rad)
lower_limits = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0]
)
upper_limits = np.array(
    [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973, 0.04, 0.04]
)

class IKSolver:
    def __init__(self, reset_joint_pos, world2robot_homo=None):
        # Configuration
        self.reset_joint_pos = np.asarray(reset_joint_pos, dtype=np.float32)
        self.world2robot_homo = world2robot_homo  # interface kept for compatibility (unused)
        # Panda joint limit arrays
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        # Default rest pose (e.g. 30% between lower and upper limits)
        self.rest_pose_default = 0.3 * (self.lower_limits + self.upper_limits)
        self.joint_ranges = self.upper_limits - self.lower_limits
        self.damping = 0.05  # damping factor for IK
        # IK iteration parameters
        self.max_iterations = 100
        self.tolerance = 1e-3  # position tolerance in meters

        # Initialize PyBullet in DIRECT mode
        self.client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        # Load the Panda URDF
        urdf_path = os.path.join(
            os.path.dirname(__file__), "data/franka_description/robots/franka_panda.urdf",
        )
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=self.client_id)
        # Find end-effector link index (link name ends with "panda_hand")
        self.end_effector_link_index = next(
            (i for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id))
             if p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)[12]
                .decode().endswith("panda_hand")),
            8  # default to 8 if not found
        )

    def solve(self, target_pose, start_joint_pos=None):
        # Ensure target_pose is numpy array
        target_pose = np.asarray(target_pose, dtype=np.float32)
        target_pos = target_pose[:3]
        # Accept either quaternion (7D) or Euler (6D) input
        if target_pose.size == 7:
            target_quat = target_pose[3:7]              # [qx, qy, qz, qw]
        else:
            target_quat = p.getQuaternionFromEuler(target_pose[3:6])  # convert euler to quat

        # Initial joint positions
        if start_joint_pos is None:
            current_joints = self.rest_pose_default.copy()
        else:
            current_joints = np.asarray(start_joint_pos, dtype=np.float32).copy()

        success = False
        position_error = None
        orientation_error = None

        for it in range(self.max_iterations):
            # Set robot to current joint positions and get end-effector state
            for j, q in enumerate(current_joints):
                p.resetJointState(self.robot_id, j, float(q), physicsClientId=self.client_id)
            link_state = p.getLinkState(self.robot_id, self.end_effector_link_index, physicsClientId=self.client_id)
            current_pos = np.array(link_state[0])    # end-effector world position
            current_orn = np.array(link_state[1])    # end-effector orientation (qx,qy,qz,qw)

            # Compute position error
            pos_err_vec = target_pos - current_pos
            position_error = np.linalg.norm(pos_err_vec)

            # Compute orientation error quaternion
            # (PyBullet returns [x,y,z,w], ensure same format for target_quat)
            # Conjugate of current orientation
            q_inv = np.array([ -current_orn[0], -current_orn[1], -current_orn[2], current_orn[3] ])
            # Quaternion multiplication Δq = q_target * q_inv:
            dq = p.getQuaternionFromEuler([0,0,0])  # placeholder, we will compute manually below
            # Manual quaternion multiplication (target_quat * q_inv):
            dq = np.array([
                target_quat[3]*q_inv[0] + target_quat[0]*q_inv[3] + target_quat[1]*q_inv[2] - target_quat[2]*q_inv[1],
                target_quat[3]*q_inv[1] + target_quat[1]*q_inv[3] + target_quat[2]*q_inv[0] - target_quat[0]*q_inv[2],
                target_quat[3]*q_inv[2] + target_quat[2]*q_inv[3] + target_quat[0]*q_inv[1] - target_quat[1]*q_inv[0],
                target_quat[3]*q_inv[3] - target_quat[0]*q_inv[0] - target_quat[1]*q_inv[1] - target_quat[2]*q_inv[2]
            ], dtype=np.float32)
            # Ensure shortest path
            if dq[3] < 0:
                dq *= -1
            # Orientation error angle
            orientation_error = 2 * math.acos(max(min(dq[3], 1.0), -1.0))
            # Orientation error vector (axis-angle form scaled by angle; for small angles ~ 2*vector part)
            if np.linalg.norm(dq[:3]) < 1e-6:
                orn_err_vec = np.zeros(3)  # no rotation needed (or very small)
            else:
                axis = dq[:3] / np.linalg.norm(dq[:3])
                orn_err_vec = axis * orientation_error

            # Check convergence
            if position_error < self.tolerance and orientation_error < self.tolerance:
                success = True
                break
            
            if len(current_joints) == 7:
                current_joints = np.concatenate((current_joints, [0.0,0.0]))
            # Compute Jacobian (linear and angular) at current joint state
            j_linear, j_angular = p.calculateJacobian(
                bodyUniqueId=self.robot_id,
                linkIndex=self.end_effector_link_index,
                localPosition=[0, 0, 0],
                objPositions=list(current_joints),
                objVelocities=[0.0]*len(current_joints),
                objAccelerations=[0.0]*len(current_joints)
            )
            # Jacobian comes as lists of length 3*DOF, convert to 3xDOF arrays
            j_linear = np.reshape(j_linear, (3, -1))
            j_angular = np.reshape(j_angular, (3, -1))
            J_full = np.vstack((j_linear, j_angular))  # 6 x N matrix

            # Solve for joint increment (damped pseudoinverse for stability)
            J_pinv = np.linalg.pinv(J_full)  # or use damped least squares if needed
            err_6x1 = np.hstack((pos_err_vec, orn_err_vec))
            delta_theta = J_pinv.dot(err_6x1)

            # Update joints
            current_joints += delta_theta
            # Optionally, clamp joint angles within limits
            current_joints = np.clip(current_joints, self.lower_limits, self.upper_limits)
        # end for

        # Prepare result
        final_joints = current_joints[:7]
        status_msg = "SUCCESS" if success else f"FAILURE: position_error={position_error:.6f}, orientation_error={orientation_error:.6f}"
        return SimpleNamespace(
            cspace_position=np.array(final_joints, dtype=np.float32),
            jpos=final_joints.tolist(),
            position_error=position_error,
            success=success,
            status=status_msg
        )


    def __del__(self):
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

# 简单自测 (simple test)
if __name__ == "__main__":
    reset_joint_pos = 0.3 * (lower_limits + upper_limits)
    solver = IKSolver(reset_joint_pos=reset_joint_pos)
    target_pos = [0.5, 0.0, 0.5]
    target_quat = p.getQuaternionFromEuler([np.pi, 0, 0])  # end-effector pointing down
    target = target_pos + list(target_quat)
    start_joint_pos = solver.rest_pose_default
    res = solver.solve(target, start_joint_pos)
    print("Solution joints:", res.cspace_position)
    print("Position error:", res.position_error)
    print("Status:", res.status)

import os
import pybullet as p
import pybullet_data

import numpy as np
from isaacgym.torch_utils import *
import torch

# ----------------------------- IK core (user‑supplied) -----------------------------
def compute_joint_angles_for_pose_iterative(
        target_pos,
        target_rot,
        initial_joint_pos,
        get_current_state_fn,
        get_jacobian_fn,
        max_iterations=50,
        tolerance=1e-3,
        damping=0.05,
        step_size=1.0,
):
    current_joint_pos = initial_joint_pos.clone()
    for _ in range(max_iterations):
        current_pos, current_rot = get_current_state_fn(current_joint_pos)

        # position & orientation error
        pos_err = target_pos - current_pos
        cc = quat_conjugate(current_rot)
        orn_err = (quat_mul(target_rot, cc))[0:3] * torch.sign((quat_mul(target_rot, cc))[3])

        # termination test
        if torch.norm(pos_err) + torch.norm(orn_err) < tolerance:
            return current_joint_pos, True

        J = get_jacobian_fn(current_joint_pos)
        dpose = torch.cat((pos_err, orn_err)).unsqueeze(-1)

        JT = J.T
        lam = torch.eye(6, device=J.device) * (damping ** 2)
        dq = (JT @ torch.inverse(J @ JT + lam) @ dpose).squeeze(-1)

        current_joint_pos += step_size * dq
    return current_joint_pos, False

# ----------------------------- Simple PyBullet wrapper -----------------------------
class IKSolver:
    def __init__(self):
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        urdf_path = os.path.join(
            os.path.dirname(__file__),
            "data/franka_description/robots/franka_panda.urdf",
        )
        self.robot = p.loadURDF(urdf_path, useFixedBase=True)

        self.joint_indices = [
            j for j in range(p.getNumJoints(self.robot))
            if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE
        ]

        # end‑effector (link named "panda_hand" in the URDF)
        self.ee_link = next(
            i for i in range(p.getNumJoints(self.robot))
            if p.getJointInfo(self.robot, i)[12].decode() == "panda_hand"
        )

    # forward kinematics: returns (pos, quat) as torch tensors
    def get_current_state(self, joint_pos_tensor):
        for idx, q in zip(self.joint_indices, joint_pos_tensor.tolist()):
            p.resetJointState(self.robot, idx, q)

        ls = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True)
        pos, orn = torch.tensor(ls[4]), torch.tensor(ls[5])
        return pos.float(), orn.float()

    
    def _expand(self, q_arm):
        q_full = [0.0]*p.getNumJoints(self.robot)  # -> 9
        for idx,val in zip(self.joint_indices, q_arm.tolist()):
            q_full[idx] = val
        return list(q_full)

    # geometric Jacobian (6×N) as torch tensor
    def get_jacobian(self, joint_pos_tensor):
        zeros = [0.0] * len(self.joint_indices)
        jac_t, jac_r = p.calculateJacobian(
            self.robot,
            self.ee_link,
            [0, 0, 0],
            self._expand(joint_pos_tensor),
            zeros,
            zeros,
        )
        J = np.vstack((jac_t, jac_r))          # 6×N
        return torch.tensor(J).float()

# ----------------------------- Demo ------------------------------------------------
if __name__ == "__main__":
    solver = IKSolver()

    # start from home posture
    q0 = torch.zeros(len(solver.joint_indices))

    # simple cartesian goal
    goal_pos = torch.tensor([0.5, 0.0, 0.3])
    goal_rot = torch.tensor([0.0, 0.0, 0.0, 1.0])     # identity quaternion

    q_sol, ok = compute_joint_angles_for_pose_iterative(
        goal_pos, goal_rot, q0,
        solver.get_current_state,
        solver.get_jacobian,
    )

    print("Converged:", ok)
    print("Joint solution (rad):", q_sol.tolist())

    p.disconnect()

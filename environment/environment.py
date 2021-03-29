import torch
import config as cfg
import environment.transformations as tra
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Encapsulates the registration environment behavior, i.e., initialization, updating the state given an action, computing
the reward for an action in the current state and, additionally, provides the expert policy's action for a given state. 
"""

ALL_STEPS = torch.FloatTensor(cfg.STEPSIZES[::-1] + [0] + cfg.STEPSIZES).to(DEVICE)
POS_STEPS = torch.FloatTensor([0] + cfg.STEPSIZES).to(DEVICE)
NUM_STEPS = len(cfg.STEPSIZES)


def init(data):
    """
    Get the initial observation, the ground-truth pose for the expert and initialize the agent's accumulator (identity).
    """
    # observation
    pcd_source, pcd_target = data['points_src'][..., :3].to(DEVICE), data['points_ref'][..., :3].to(DEVICE)
    B = pcd_source.shape[0]

    # GT (for expert)
    pose_target = torch.eye(4, device=DEVICE).repeat(B, 1, 1)
    pose_target[:, :3, :] = data['transform_gt']

    # initial estimates (identity, for student)
    pose_source = torch.eye(4, device=DEVICE).repeat(B, 1, 1)

    return pcd_source, pcd_target, pose_source, pose_target


def _action_to_step(axis_actions):
    """
    Convert action ids to sign and step size.
    """
    step = ALL_STEPS[axis_actions]
    sign = ((axis_actions - NUM_STEPS >= 0).float() - 0.5) * 2
    return sign, step


def step(source, actions, pose_source, disentangled=True):
    """
    Update the state (source and accumulator) using the given actions.
    """
    actions_t, actions_r = actions[:, 0], actions[:, 1]
    indices = torch.arange(source.shape[0]).unsqueeze(0)

    # actions to transformations
    steps_t = torch.zeros((actions.shape[0], 3), device=DEVICE)
    steps_r = torch.zeros((actions.shape[0], 3), device=DEVICE)
    for i in range(3):
        sign, step = _action_to_step(actions_t[:, i])
        steps_t[indices, i] = step * sign

        sign, step = _action_to_step(actions_r[:, i])
        steps_r[indices, i] = step * sign

    # accumulate transformations
    if disentangled:  # eq. 7 in paper
        pose_source[:, :3, :3] = tra.euler_angles_to_matrix(steps_r, 'XYZ') @ pose_source[:, :3, :3]
        pose_source[:, :3, 3] += steps_t
    else:  # concatenate 4x4 matrices (eq. 5 in paper)
        pose_update = torch.eye(4, device=DEVICE).repeat(pose_source.shape[0], 1, 1)
        pose_update[:, :3, :3] = tra.euler_angles_to_matrix(steps_r, 'XYZ')
        pose_update[:, :3, 3] = steps_t

        pose_source = pose_update @ pose_source

    # update source with the accumulated transformation
    new_source = tra.apply_trafo(source, pose_source, disentangled)

    return new_source, pose_source


def expert(pose_source, targets, mode='steady'):
    """
    Get the expert action in the current state.
    """
    # compute delta, eq. 10 in paper
    delta_t = targets[:, :3, 3] - pose_source[:, :3, 3]
    delta_R = targets[:, :3, :3] @ pose_source[:, :3, :3].transpose(2, 1)  # global accumulator
    delta_r = tra.matrix_to_euler_angles(delta_R, 'XYZ')

    def _get_axis_action(axis_delta, mode='steady'):
        lower_idx = (torch.bucketize(torch.abs(axis_delta), POS_STEPS) - 1).clamp(0, NUM_STEPS)
        if mode == 'steady':
            nearest_idx = lower_idx
        elif mode == 'greedy':
            upper_idx = (lower_idx + 1).clamp(0, NUM_STEPS)
            lower_dist = torch.abs(torch.abs(axis_delta) - POS_STEPS[lower_idx])
            upper_dist = torch.abs(POS_STEPS[upper_idx] - torch.abs(axis_delta))
            nearest_idx = torch.where(lower_dist < upper_dist, lower_idx, upper_idx)
        else:
            raise ValueError

        # -- step idx to action
        axis_action = nearest_idx  # [0, num_steps] -- 0 = NOP
        axis_action[axis_delta < 0] *= -1  # [-num_steps, num_steps + 1] -- 0 = NOP
        axis_action += NUM_STEPS  # [0, 2 * num_steps + 1 -- num_steps = NOP

        return axis_action[:, None, None]

    # find bounds per axis s.t. b- <= d <= b+
    action_t = torch.cat([_get_axis_action(delta_t[:, i], mode) for i in range(3)], dim=2)
    action_r = torch.cat([_get_axis_action(delta_r[:, i], mode) for i in range(3)], dim=2)
    action = torch.cat([action_t, action_r], dim=1)

    return action


def reward_step(current_pcd_source, gt_pcd_source, prev_chamfer_dist=None):
    """
    Compute the dense step reward for the updated state.
    """
    dist = torch.min(tra.square_distance(current_pcd_source, gt_pcd_source), dim=-1)[0]
    chamfer_dist = torch.mean(dist, dim=1).view(-1, 1, 1)

    if prev_chamfer_dist is not None:
        better = (chamfer_dist < prev_chamfer_dist).float() * 0.5
        same = (chamfer_dist == prev_chamfer_dist).float() * 0.1
        worse = (chamfer_dist > prev_chamfer_dist).float() * 0.6

        reward = better - worse - same
        return reward, chamfer_dist
    else:
        return torch.zeros_like(chamfer_dist), chamfer_dist


def reward_goal(pose_source, targets, goal_th=2):
    """
    Compute the sparse goal reward for the updated state.
    """
    # SO(3) version, eq. 10 in paper
    delta_t = targets[:, :3, 3] - pose_source[:, :3, 3]
    delta_R = pose_source[:, :3, :3].transpose(2, 1) @ targets[:, :3, :3]
    delta_r = tra.matrix_to_euler_angles(delta_R, 'XYZ')

    delta_t = torch.abs(delta_t).view(-1, 3)
    delta_r = torch.abs(delta_r).view(-1, 3)
    deltas = torch.cat([delta_t, delta_r], dim=1).view(-1, 6, 1)

    # in goal?
    reward = (deltas < cfg.STEPSIZES[goal_th]).float()
    # take mean over dimensions
    return reward.mean(dim=1)[:, None, :]


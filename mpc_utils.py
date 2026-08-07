# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__, force=True)


def l1(a, b):
    return torch.mean(torch.abs(a - b), dim=-1)

def l2(a, b):
    return torch.mean((a - b) ** 2, dim=-1)


def round_small_elements(tensor, threshold):
    mask = torch.abs(tensor) < threshold
    new_tensor = tensor.clone()
    new_tensor[mask] = 0
    return new_tensor


def normalize_loss(loss):
    scale = loss.std(unbiased=False).clamp_min(1e-6)
    return (loss - loss.mean()) / scale


def weighted_elite_stats(actions, loss, indices, temperature=0.25):

    # select top-k actions and their corresponding losses
    selected_actions = actions[indices]
    selected_loss = loss[indices]

    # normalise the selected loss to have zero mean and unit variance
    # selected_loss = selected_loss - selected_loss.mean() / selected_loss.std(unbiased=False).clamp_min(1e-6)

    weights = torch.softmax(-selected_loss /temperature, dim=0)
    view_shape = (weights.size(0),) + (1,) * (selected_actions.dim() - 1)
    weights = weights.view(view_shape)

    mean = (selected_actions * weights).sum(dim=0)
    var = ((selected_actions - mean) ** 2 * weights).sum(dim=0)
    return mean, torch.sqrt(var.clamp_min(1e-8))


def compute_new_pose(pose, action):
    """
    :param pose: [B, T=1, 7]
    :param action: [B, T=1, 7]
    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]


def poses_to_diff(start, end):
    """
    :param start: [7]
    :param end: [7]
    """
    try:
        start = start.numpy()
        end = end.numpy()
    except Exception:
        pass

    # --

    s_xyz = start[:3]
    e_xyz = end[:3]
    xyz_diff = e_xyz - s_xyz

    # --

    s_thetas = start[3:6]
    e_thetas = end[3:6]
    s_rotation = Rotation.from_euler("xyz", s_thetas, degrees=False).as_matrix()
    e_rotation = Rotation.from_euler("xyz", e_thetas, degrees=False).as_matrix()
    rotation_diff = e_rotation @ s_rotation.T
    theta_diff = Rotation.from_matrix(rotation_diff).as_euler("xyz", degrees=False)

    # --

    s_gripper = start[-1:]
    e_gripper = end[-1:]
    gripper_diff = e_gripper - s_gripper

    action = np.concatenate([xyz_diff, theta_diff, gripper_diff], axis=0)
    return torch.from_numpy(action)


def cem(
    context_obs,
    context_frame,
    context_pose,
    goal_frame_side,
    goal_frame_wrist,
    world_model,
    rollout=1,
    cem_steps=100,
    momentum_mean=0.25,
    momentum_mean_rot=0.25,
    momentum_std=0.95,
    momentum_std_rot=0.85,
    momentum_mean_gripper=0.15,
    momentum_std_gripper=0.15,
    samples=100,
    topk=10,
    verbose=False,
    maxnorm=0.05,
    maxrotnorm=0.314,
    axis={},
    objective='l1',
    warm_starting=False,
    close_gripper=None,
    prev_action=None,
    generator=None,
    log=False,
    elite_temperature=0.5,
):
    """
    :param context_obs: {"left": [H, W, 3], "wrist": [H, W, 3]}
    :param context_frame: {"left": [B=1, T=1, HW, D], "wrist": [B=1, T=1, HW, D]}
    :param goal_frame_side: [B=1, T=1, HW, D]
    :param goal_frame_wrist: [B=1, T=1, HW, D]
    :param world_model: f(context_frame, action) -> next_frame [B, 1, HW, D]
    :return: [B=1, rollout, 7] an action trajectory over rollout horizon

    Cross-Entropy Method
    -----------------------
    1. for rollout horizon:
    1.1. sample several actions
    1.2. compute next states using WM
    3. compute similarity of final states to goal_frames
    4. select topk samples and update mean and std using topk action trajs
    5. choose final action to be mean of distribution
    """

    def sample_action_traj():
        """Sample several action trajectories"""
        # pass h_current, s_current
        action_traj = None
        frame_traj_side, frame_traj_wrist = context_frame["left"], context_frame["wrist"]
        pose_traj = context_pose

        for h in range(rollout):

            action_samples = torch.randn(
                samples,
                mean.size(1),
                device=mean.device,
                generator=generator,
            ) * std[h] + mean[h]
            action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm)
            action_samples[:, 3:6] = torch.clip(action_samples[:, 3:6], min=-maxrotnorm, max=maxrotnorm)
            action_samples[:, -1:] = torch.clip(action_samples[:, -1:], min=-0.75, max=0.75)
            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]

            action_samples = action_samples[:, None]
            if close_gripper is not None and h >= close_gripper:
                action_samples[:, :, -1] = 1.0

            # concatenate action to ACTION TRAJ
            action_traj = (
                torch.cat([action_traj, action_samples], dim=1) if action_traj is not None \
                                                        else action_samples
            )

            # predict next observation embedding and pose with vjepa-ac model
            next_frame, next_pose = world_model(
                context_obs,
                {"left": frame_traj_side, "wrist": frame_traj_wrist},
                action_traj,
                pose_traj,
                plot=False,
            )
            next_frame_side = next_frame["left"]
            next_frame_wrist = next_frame["wrist"]

            # concatenate to trajectory
            frame_traj_side = torch.cat([frame_traj_side, next_frame_side], dim=1)
            frame_traj_wrist = torch.cat([frame_traj_wrist, next_frame_wrist], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

        return action_traj, frame_traj_side, frame_traj_wrist

    def select_topk_action_traj(final_state_side, goal_state_side,
                                final_state_wrist, goal_state_wrist, actions):
        """Select separate elite sets for action subspaces.

        The side camera is used to rank candidates for xyz translation, while
        the wrist camera is used to rank candidates for rpy/gripper. Both losses
        are computed from the same full-action rollouts.
        """
        loss_side = l1(final_state_side.flatten(1), goal_state_side.flatten(1))
        loss_wrist = l2(final_state_wrist.flatten(1), goal_state_wrist.flatten(1))

        rank_loss_side = normalize_loss(loss_side) 
        rank_loss_wrist = normalize_loss(loss_wrist) 

        loss = 0.5 * rank_loss_side + 0.5 * rank_loss_wrist

        indices = loss.topk(topk, largest=False).indices

        mean, std = weighted_elite_stats(
            actions, loss, indices, temperature=elite_temperature
        )
        return mean, std

    # INPUTS
    context_obs_side = context_obs["left"].repeat(samples, 1, 1, 1) # Reshape to [S, 3, H, W]
    context_frame_side = context_frame["left"].repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    goal_frame_side = goal_frame_side.repeat(samples, 1, 1, 1)  # Reshape to [S, 1, HW, D]
    
    context_obs_wrist = context_obs["wrist"].repeat(samples, 1, 1, 1) 
    context_frame_wrist = context_frame["wrist"].repeat(samples, 1, 1, 1) 
    goal_frame_wrist = goal_frame_wrist.repeat(samples, 1, 1, 1)  
    context_obs = {"left": context_obs_side, "wrist": context_obs_wrist}
    context_frame = {"left": context_frame_side, "wrist": context_frame_wrist}
    
    context_pose = context_pose.repeat(samples, 1, 1)  # Reshape to [S, 1, 7]

    # Current estimate of the mean/std of distribution over action trajectories
    mean = torch.zeros((rollout, 7), device=context_frame["left"].device)

    std = torch.cat(
        [
            torch.ones((rollout, 3), device=context_frame["left"].device) * maxnorm,
            torch.ones((rollout, 3), device=context_frame["left"].device) * maxrotnorm,
            # gripper still needs std upto 1.0 to explore open/close actions
            torch.ones((rollout, 1), device=context_frame["left"].device),
        ],
        dim=-1,
    )

    if warm_starting: 
        if prev_action is not None:
            # use provided prev_action for first step mean; 
            mean[:-1] = prev_action

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    for step in tqdm(range(cem_steps), disable=True):

        # samples action traj and predicts new frame embeddings
        action_traj, frame_traj_side, frame_traj_wrist = sample_action_traj()

        # compute distance between final frame and each goal frame
        (
            mean_selected_actions,
            std_selected_actions,
        ) = select_topk_action_traj(
            final_state_side=frame_traj_side[:, -1],
            goal_state_side=goal_frame_side,
            final_state_wrist=frame_traj_wrist[:, -1],
            goal_state_wrist=goal_frame_wrist,
            actions=action_traj,
        )

        # -- Update new sampling mean and std based on the top-k samples
        mean = torch.cat(
            [
                mean_selected_actions[..., :3] * (1.0 - momentum_mean) + mean[..., :3] * momentum_mean,
                mean_selected_actions[..., 3:6] * (1.0 - momentum_mean_rot) + mean[..., 3:6] * momentum_mean_rot,
                mean_selected_actions[..., -1:] * (1.0 - momentum_mean_gripper)
                + mean[..., -1:] * momentum_mean_gripper,
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                std_selected_actions[..., :3] * (1.0 - momentum_std) + std[..., :3] * momentum_std,
                std_selected_actions[..., 3:6] * (1.0 - momentum_std_rot) + std[..., 3:6] * momentum_std_rot,
                std_selected_actions[..., -1:] * (1.0 - momentum_std_gripper) + std[..., -1:] * momentum_std_gripper,
            ],
            dim=-1,
        )

        # logger.info(f"new mean: {mean.sum(dim=0)} {std.sum(dim=0)}")

    new_action = torch.cat(
        [
            mean[..., :3],
            round_small_elements(mean[..., 3:6], 0.05),
            round_small_elements(mean[..., -1:], 0.15),
        ],
        dim=-1,
    )[None, :]

    if log:
        # Roll out the final selected action trajectory once to trigger optional
        # reconstruction logging/debug visualization inside the world model.
        frame_traj_side, frame_traj_wrist, pose_traj = context_frame["left"][:1], context_frame["wrist"][:1], context_pose[:1]
        context_obs_single = {"left": context_obs["left"][:1], "wrist": context_obs["wrist"][:1]}
        for h in range(rollout):
            next_frame, next_pose = world_model(
                context_obs_single,
                {"left": frame_traj_side, "wrist": frame_traj_wrist},
                new_action[:, : h + 1],
                pose_traj,
                h == (rollout - 1),
            )
            next_frame_side = next_frame["left"]
            next_frame_wrist = next_frame["wrist"]
            frame_traj_side = torch.cat([frame_traj_side, next_frame_side], dim=1)
            frame_traj_wrist = torch.cat([frame_traj_wrist, next_frame_wrist], dim=1)
            pose_traj = torch.cat([pose_traj, next_pose], dim=1)

    return new_action, mean

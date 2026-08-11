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


def relative_goal_loss(candidate_loss, reference_loss, eps=1e-6):
    """Normalize candidate goal loss by a fixed predicted no-op goal loss.

    The reference is computed once at the start of each MPC/CEM solve by
    rolling the same world model forward for the same horizon with zero
    actions. A value of 1.0 therefore means the candidate is predicted to be
    as far from the goal as doing nothing, values below 1.0 are better than
    doing nothing, and values above 1.0 are worse than doing nothing.
    """
    return candidate_loss / reference_loss.clamp_min(eps)


def weighted_elite_stats(actions, loss, indices, temperature=0.25):

    # Select top-k actions and corresponding losses.
    selected_actions = actions[indices]
    selected_loss = loss[indices]

    # Normalize the spread of losses within the selected top-k set.
    loss_std = selected_loss.std(
        unbiased=False
    ).clamp_min(1e-6)

    normalized_loss = (
        selected_loss - selected_loss.min()
    ) / loss_std

    weights = torch.softmax(
        -normalized_loss / temperature,
        dim=0,
    )

    # # Effective number of selected trajectories contributing.
    # ess = 1.0 / weights.pow(2).sum()

    # logger.info(
    #     f"Loss-weighted top-k: "
    #     f"loss_min={selected_loss.min().item():.4f} "
    #     f"loss_max={selected_loss.max().item():.4f} "
    #     f"loss_std={loss_std.item():.6f} "
    #     f"ESS={ess.item():.2f}/{weights.numel()} "
    #     f"w_max={weights.max().item():.3f}"
    # )

    view_shape = (
        weights.size(0),
    ) + (1,) * (selected_actions.dim() - 1)

    weights = weights.view(view_shape)

    mean = (
        selected_actions * weights
    ).sum(dim=0)

    var = (
        (selected_actions - mean) ** 2 * weights
    ).sum(dim=0)

    return mean, torch.sqrt(
        var.clamp_min(1e-8)
    )


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
    matrices = [
        Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
        for theta in thetas
    ]
    delta_matrices = [
        Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
        for theta in delta_thetas
    ]
    angle_diff = [
        delta_matrices[t] @ matrices[t] for t in range(len(matrices))
    ]
    angle_diff = [
        Rotation.from_matrix(mat).as_euler("xyz", degrees=False)
        for mat in angle_diff
    ]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate(
        [new_xyz, new_angle, new_closedness],
        axis=-1,
    )
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
    s_rotation = Rotation.from_euler(
        "xyz",
        s_thetas,
        degrees=False,
    ).as_matrix()
    e_rotation = Rotation.from_euler(
        "xyz",
        e_thetas,
        degrees=False,
    ).as_matrix()
    rotation_diff = e_rotation @ s_rotation.T
    theta_diff = Rotation.from_matrix(
        rotation_diff
    ).as_euler("xyz", degrees=False)

    # --

    s_gripper = start[-1:]
    e_gripper = end[-1:]
    gripper_diff = e_gripper - s_gripper

    action = np.concatenate(
        [xyz_diff, theta_diff, gripper_diff],
        axis=0,
    )
    return torch.from_numpy(action)


def cem(
    context_obs,
    context_frame,
    context_pose,
    goal_frame,
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
    objective="l1",
    warm_starting=False,
    close_gripper=None,
    prev_action=None,
    generator=None,
    log=False,
    elite_temperature=0.25,
    relative_loss_eps=1e-6,
):
    """
    :param context_obs: [H, W, 3]
    :param context_frame: [B=1, T=1, HW, D]
    :param goal_frame: [B=1, T=1, HW, D]
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
        """Sample several action trajectories."""
        action_traj = None
        frame_traj = context_frame
        pose_traj = context_pose

        for h in range(rollout):

            action_samples = (
                torch.randn(samples, mean.size(1), device=mean.device, generator=generator,) * std[h] + mean[h]
            )

            action_samples[:, :3] = torch.clip(action_samples[:, :3], min=-maxnorm, max=maxnorm,
            )
            action_samples[:, 3:6] = torch.clip(
                action_samples[:, 3:6],
                min=-maxrotnorm,
                max=maxrotnorm,
            )
            action_samples[:, -1:] = torch.clip(
                action_samples[:, -1:],
                min=-0.75,
                max=0.75,
            )

            for ax in axis.keys():
                action_samples[:, ax] = axis[ax]

            action_samples = action_samples[:, None]

            if close_gripper is not None and h >= close_gripper:
                action_samples[:, :, -1] = 1.0

            action_traj = (
                torch.cat(
                    [action_traj, action_samples],
                    dim=1,
                )
                if action_traj is not None
                else action_samples
            )

            next_frame, next_pose = world_model(
                context_obs,
                frame_traj,
                action_traj,
                pose_traj,
                plot=False,
            )

            frame_traj = torch.cat(
                [frame_traj, next_frame],
                dim=1,
            )
            pose_traj = torch.cat(
                [pose_traj, next_pose],
                dim=1,
            )

        return (
            action_traj,
            frame_traj,
        )

    def rollout_zero_action_reference(
        reference_context_obs,
        reference_context_frame,
        reference_context_pose,
    ):
        """Roll out the world model for the planning horizon with zero actions.

        This produces a reference on the same predictor-output distribution
        and at the same rollout horizon as the CEM candidates.

        close_gripper and fixed-axis overrides are intentionally not applied:
        this is a true no-op action trajectory.
        """

        action_traj = None
        frame_traj = reference_context_frame.unsqueeze(0)
        pose_traj = reference_context_pose.unsqueeze(0)

        for _ in range(rollout):

            zero_action = torch.zeros(
                (1, 1, 7),
                device=reference_context_pose.device,
                dtype=reference_context_pose.dtype,
            )

            action_traj = (
                torch.cat(
                    [action_traj, zero_action],
                    dim=1,
                )
                if action_traj is not None
                else zero_action
            )

            next_frame, next_pose = world_model(
                reference_context_obs,
                frame_traj,
                action_traj,
                pose_traj,
                plot=False,
            )

            frame_traj = torch.cat(
                [frame_traj, next_frame],
                dim=1,
            )
            pose_traj = torch.cat(
                [pose_traj, next_pose],
                dim=1,
            )

        return frame_traj[:, -1]

    def select_topk_action_traj(
        reference_loss,
        final_state,
        goal_state,
        actions,
    ):
        """Select elites using scale-normalized single-view cost."""

        absolute_loss = l1(
            final_state.flatten(1),
            goal_state.flatten(1),
        )


        loss = relative_goal_loss(
            absolute_loss,
            reference_loss,
            eps=relative_loss_eps,
        )

        if verbose:

            best_idx = loss.argmin()

            logger.info(
                f"CEM best candidate: "
                f"combined={loss[best_idx].item()} "
            )

        indices = loss.topk(
            topk,
            largest=False,
        ).indices

        # selected_actions = actions[indices]
        # return selected_actions
    
        # now obtain mean and var deom elite samples using weighted softmax
        mean, std = weighted_elite_stats(
            actions,
            loss,
            indices,
            temperature=elite_temperature,
        )

        return mean, std



    # ---------------------------------------------------------
    # Diagnostic only:
    # current encoded latent -> goal latent.
    #
    # ---------------------------------------------------------

    # encoded_current_loss = l1(
    #     context_frame.flatten(1),
    #     goal_frame.flatten(1),
    # ).mean().detach()

    # ---------------------------------------------------------
    # Zero-action predictor reference:
    # predicted no-op latent -> goal latent.
    #
    # This uses the same world model and the same planning
    # horizon as the candidate trajectories.
    # ---------------------------------------------------------

    zero_final = (
        rollout_zero_action_reference(
            context_obs,
            context_frame,
            context_pose,
        )
    )

    reference_loss = l1(
        zero_final.flatten(1),
        goal_frame.flatten(1),
    ).mean().detach()

    if verbose:

        logger.info(
            f"Zero-action predicted reference goal losses: {reference_loss.item()}"
        )

    # ---------------------------------------------------------
    # Expand MPC inputs over candidate samples.
    # ---------------------------------------------------------

    context_obs = context_obs.repeat(
        samples,
        1,
        1,
        1,
    )

    context_frame = context_frame.repeat(
        samples,
        1,
        1,
        1,
    )

    goal_frame = goal_frame.repeat(
        samples,
        1,
        1,
        1,
    )

    context_pose = context_pose.repeat(
        samples,
        1,
        1,
    )

    # ---------------------------------------------------------
    # Initial proposal distribution.
    # ---------------------------------------------------------

    mean = torch.zeros(
        (rollout, 7),
        device=context_frame.device,
    )

    std = torch.cat(
        [
            torch.ones(
                (rollout, 3),
                device=context_frame.device,
            )
            * maxnorm,
            torch.ones(
                (rollout, 3),
                device=context_frame.device,
            )
            * maxrotnorm,
            # gripper still needs std up to 1.0
            # to explore open/close actions
            torch.ones(
                (rollout, 1),
                device=context_frame.device,
            ) * 0.75,
        ],
        dim=-1,
    )

    if warm_starting:
        if prev_action is not None:
            mean[:-1] = prev_action

    for ax in axis.keys():
        mean[:, ax] = axis[ax]

    # ---------------------------------------------------------
    # CEM optimization.
    # ---------------------------------------------------------

    for step in tqdm(range(cem_steps), disable=True):

        if verbose:

            logger.info(
                f"CEM step={step} proposal BEFORE sampling: "
                f"mean_norm_xyz="
                f"{mean[..., :3].norm(dim=-1).mean().item()} "
                f"mean_norm_rot="
                f"{mean[..., 3:6].norm(dim=-1).mean().item()} "
                f"mean_abs_grip="
                f"{mean[..., -1:].abs().mean().item()} "
                f"std_xyz="
                f"{std[..., :3].mean().item()} "
                f"std_rot="
                f"{std[..., 3:6].mean().item()} "
                f"std_grip="
                f"{std[..., -1:].mean().item()}"
            )

        action_traj, frame_traj = (
            sample_action_traj()
        )

        if verbose:

            empirical_std = action_traj.std(
                dim=0,
                unbiased=False,
            )

            logger.info(
                f"CEM step={step} sampled empirical std: "
                f"xyz={empirical_std[..., :3].mean().item()} "
                f"rot={empirical_std[..., 3:6].mean().item()} "
                f"grip={empirical_std[..., -1:].mean().item()}"
            )

        mean_selected_actions, std_selected_actions = select_topk_action_traj(
            reference_loss=reference_loss,
            final_state=frame_traj[:, -1],
            goal_state=goal_frame,
            actions=action_traj,
        )

        # mean_selected_actions = selected_actions.mean(
        #     dim=0,
        # )
        # std_selected_actions = selected_actions.std(
        #     dim=0,
        # )

        mean = torch.cat(
            [
                (
                    mean_selected_actions[..., :3]
                    * (1.0 - momentum_mean)
                    + mean[..., :3] * momentum_mean
                ),
                (
                    mean_selected_actions[..., 3:6]
                    * (1.0 - momentum_mean_rot)
                    + mean[..., 3:6] * momentum_mean_rot
                ),
                (
                    mean_selected_actions[..., -1:]
                    * (1.0 - momentum_mean_gripper)
                    + mean[..., -1:]
                    * momentum_mean_gripper
                ),
            ],
            dim=-1,
        )

        std = torch.cat(
            [
                (
                    std_selected_actions[..., :3]
                    * (1.0 - momentum_std)
                    + std[..., :3] * momentum_std
                ),
                (
                    std_selected_actions[..., 3:6]
                    * (1.0 - momentum_std_rot)
                    + std[..., 3:6] * momentum_std_rot
                ),
                (
                    std_selected_actions[..., -1:]
                    * (1.0 - momentum_std_gripper)
                    + std[..., -1:]
                    * momentum_std_gripper
                ),
            ],
            dim=-1,
        )

        if verbose:

            logger.info(
                f"CEM step={step} proposal AFTER elite update: "
                f"mean_norm_xyz="
                f"{mean[..., :3].norm(dim=-1).mean().item()} "
                f"mean_norm_rot="
                f"{mean[..., 3:6].norm(dim=-1).mean().item()} "
                f"mean_abs_grip="
                f"{mean[..., -1:].abs().mean().item()} "
                f"std_xyz="
                f"{std[..., :3].mean().item()} "
                f"std_rot="
                f"{std[..., 3:6].mean().item()} "
                f"std_grip="
                f"{std[..., -1:].mean().item()}"
            )

    # ---------------------------------------------------------
    # Return the mean of the final proposal.
    # ---------------------------------------------------------

    new_action = torch.cat(
        [
            mean[..., :3],
            round_small_elements(mean[..., 3:6], 0.05),
            mean[..., -1:],
        ],
        dim=-1,
    )

    if log:

        frame_traj = context_frame[:1]
        pose_traj = context_pose[:1]

        for h in range(rollout):

            next_frame, next_pose = world_model(
                context_obs[:1],
                frame_traj,
                new_action[:, : h + 1],
                pose_traj,
                h == (rollout - 1),
            )


            frame_traj = torch.cat(
                [frame_traj, next_frame],
                dim=1,
            )

            pose_traj = torch.cat(
                [pose_traj, next_pose],
                dim=1,
            )

    logger.info(
    f"Executed action: "
    f"xyz={new_action[0, :3].tolist()} "
    f"rot={new_action[0, 3:6].tolist()} "
    f"grip={new_action[0, 6].item()}"
    )

    return new_action

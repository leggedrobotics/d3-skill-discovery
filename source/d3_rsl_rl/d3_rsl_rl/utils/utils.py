# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
import torch

import git


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6], | [ [False, False, False, True, False, True],
                 [b1, b2 | b3, b4, b5 | b6] |   [False, True, False, False, True, True],
                ]                           | ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the input has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # add at least one full length trajectory
    # trajectories = trajectories + (torch.zeros(tensor.shape[0], tensor.shape[-1], device=tensor.device),)
    trajectories = trajectories + (torch.zeros((tensor.shape[0],) + tensor.shape[2:], device=tensor.device),)

    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
            t = repo.head.commit.tree
        except Exception:
            # skip if not a git repository
            continue
        # get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
        with open(diff_file_name, "w", encoding="utf-8") as f:
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths


def flatten_batch(
    tensor: torch.Tensor | dict[str, torch.Tensor], data_dim: torch.Size | dict[str, torch.Size]
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Flatten the batch dimensions of a tensor to a single dimension.
    Args:
        tensor: The tensor to flatten. dim is [*batch_dim, *data_dim].
        data_dim: The shape of the data dimensions or data without the batch dimensions.
    Returns:
        The flattened tensor of shape [prod(batch_dim), *data_dim].
    """
    if isinstance(tensor, dict):
        return {key: tensor[key].view(-1, *data_dim[key]) for key in tensor}
    else:
        return tensor.view(-1, *data_dim)


def unflatten_batch(
    tensor: torch.Tensor | dict[str, torch.Tensor], batch_dim: torch.Size
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    Unflatten the batch dimensions of a tensor to a single dimension.
    Args:
        tensor: The tensor to unflatten. dim is [prod(batch_dim), *data_dim].
        batch_dim: The shape of the batch dimensions.
    Returns:
        The unflattened tensor of shape [*batch_dim, *data_dim].
    """
    if isinstance(tensor, dict):
        return {key: tensor[key].view(*batch_dim, *tensor[key].shape[1:]) for key in tensor}
    else:
        return tensor.view(*batch_dim, *tensor.shape[1:])


def extract_batch_shape(
    tensor: torch.Tensor | dict[str, torch.Tensor], data_dim: torch.Size | dict[str, torch.Size]
) -> torch.Size:
    """Extracts the batch shape from a tensor or dictionary of tensors.

    Args:
        tensor: The tensor or dictionary of tensors to extract the batch shape from.
        data_dim: The shape of the data dimensions or data without the batch dimensions.
    """
    if isinstance(tensor, dict) and isinstance(data_dim, dict):
        return next(iter(tensor.values())).shape[: -len(next(iter(data_dim.values())))]
    else:
        return tensor.shape[: -len(data_dim)]


def to_device(
    data: torch.Tensor | dict[str, torch.Tensor], device: torch.device | str
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Move tensor or dictionary of tensors to the specified device."""
    if isinstance(data, dict):
        return {key: tensor.to(device) for key, tensor in data.items()}
    else:
        return data.to(device)


def detach(data: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Detach tensor or dictionary of tensors."""
    if isinstance(data, dict):
        return {key: tensor.detach() for key, tensor in data.items()}
    else:
        return data.detach()


def mean_gradient_norm(model: torch.nn.Module) -> float:
    """
    Calculates the mean of the L2 norms of the gradients of all trainable parameters in the given model.

    Args:
        model (nn.Module): The model from which to collect gradients.

    Returns:
        float: The mean of the L2 norms of the gradients.
    """
    # Initialize variables to store the sum of norms and the count of parameters
    total_norm = 0.0
    count = 0

    # Iterate through all parameters in the model
    for param in model.parameters():
        if param.grad is not None:
            # Compute the L2 norm of the gradient
            param_norm = param.grad.data.norm(2).item()
            # Sum up the norms
            total_norm += param_norm
            # Increment the count
            count += 1

    # Calculate the mean of the norms
    if count > 0:
        mean_norm = total_norm / count
    else:
        mean_norm = 0.0  # If no gradients are found, return 0

    return mean_norm


def is_valid(data: torch.Tensor | dict[str, torch.Tensor]) -> bool:
    """Check if the data is valid (not NaN or Inf)."""
    if isinstance(data, dict):
        return all(torch.isfinite(tensor).all() for tensor in data.values())
    else:
        return torch.isfinite(data).all().item()

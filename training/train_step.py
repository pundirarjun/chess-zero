"""GPU-resident replay-buffer training utilities."""

from __future__ import annotations

import numpy as np
import torch

from training.trainer import train_one_batch


def build_gpu_replay(replay_buffer, device):
    """Materialize the replay buffer on CUDA once.

    Returns ``(states, policies, values)``. The returned tensors remain on the
    supplied device and are sampled directly there during training.
    """
    if device.type != "cuda":
        raise ValueError("build_gpu_replay requires a CUDA device")
    if len(replay_buffer) == 0:
        raise ValueError("Replay buffer is empty")

    # One CPU-side packing operation followed by one asynchronous bulk transfer.
    states_np = np.asarray([s[0] for s in replay_buffer], dtype=np.float32)
    policies_np = np.asarray([s[1] for s in replay_buffer], dtype=np.float32)
    values_np = np.asarray([s[2] for s in replay_buffer], dtype=np.float32)

    states = torch.from_numpy(states_np).pin_memory().to(device, non_blocking=True)
    policies = torch.from_numpy(policies_np).pin_memory().to(device, non_blocking=True)
    values = torch.from_numpy(values_np).pin_memory().to(device, non_blocking=True)
    states = states.contiguous(memory_format=torch.channels_last)
    return states, policies, values


def train_from_gpu_replay(
    model,
    optimizer,
    gpu_replay,
    batch_size=128,
    training_steps=10,
):
    """Train by sampling entirely on CUDA after replay is resident there."""
    if batch_size <= 0 or training_steps <= 0:
        raise ValueError("batch_size and training_steps must be greater than zero")

    states_all, policies_all, values_all = gpu_replay
    device = states_all.device
    if device.type != "cuda":
        raise ValueError("gpu_replay must be on CUDA")
    n = states_all.shape[0]
    if n < batch_size:
        raise ValueError(f"Replay buffer contains {n} samples, but batch size is {batch_size}.")

    scaler = torch.amp.GradScaler("cuda")
    totals = [0.0, 0.0, 0.0]

    for step in range(1, training_steps + 1):
        indices = torch.randint(0, n, (batch_size,), device=device)
        states = states_all.index_select(0, indices)
        policies = policies_all.index_select(0, indices)
        values = values_all.index_select(0, indices)

        tl, pl, vl = train_one_batch(
            model,
            optimizer,
            states,
            policies,
            values,
            scaler=scaler,
        )
        totals[0] += tl
        totals[1] += pl
        totals[2] += vl

        if step == 1 or step % 10 == 0 or step == training_steps:
            print(
                f"Step {step}/{training_steps}: "
                f"Total={tl:.6f} | Policy={pl:.6f} | Value={vl:.6f}"
            )

    return {
        "average_total_loss": totals[0] / training_steps,
        "average_policy_loss": totals[1] / training_steps,
        "average_value_loss": totals[2] / training_steps,
    }


def train_from_replay_buffer(model, optimizer, replay_buffer, batch_size=32, training_steps=10):
    """Backward-compatible entry point.

    CUDA uses the GPU-resident path; CPU keeps the original implementation.
    """
    device = next(model.parameters()).device
    if device.type == "cuda":
        gpu_replay = build_gpu_replay(replay_buffer, device)
        return train_from_gpu_replay(model, optimizer, gpu_replay, batch_size, training_steps)

    if batch_size <= 0 or training_steps <= 0:
        raise ValueError("batch_size and training_steps must be greater than zero")
    if len(replay_buffer) < batch_size:
        raise ValueError(f"Replay buffer contains {len(replay_buffer)} samples, but batch size is {batch_size}.")

    scaler = None
    total_losses, policy_losses, value_losses = [], [], []
    for _ in range(training_steps):
        samples = replay_buffer.sample(batch_size)
        states = torch.from_numpy(np.asarray([s[0] for s in samples], dtype=np.float32)).to(device)
        policies = torch.from_numpy(np.asarray([s[1] for s in samples], dtype=np.float32)).to(device)
        values = torch.from_numpy(np.asarray([s[2] for s in samples], dtype=np.float32)).to(device)
        tl, pl, vl = train_one_batch(model, optimizer, states, policies, values, scaler=scaler)
        total_losses.append(tl); policy_losses.append(pl); value_losses.append(vl)
    return {"total_loss": total_losses, "policy_loss": policy_losses, "value_loss": value_losses}

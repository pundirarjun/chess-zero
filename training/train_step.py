"""GPU-efficient replay-buffer training."""

from __future__ import annotations

import numpy as np
import torch

from training.trainer import train_one_batch


def train_from_replay_buffer(model, optimizer, replay_buffer, batch_size=32, training_steps=10):
    if batch_size <= 0 or training_steps <= 0:
        raise ValueError("batch_size and training_steps must be greater than zero")
    if len(replay_buffer) < batch_size:
        raise ValueError(f"Replay buffer contains {len(replay_buffer)} samples, but batch size is {batch_size}.")

    device = next(model.parameters()).device
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    total_losses, policy_losses, value_losses = [], [], []

    for _ in range(training_steps):
        samples = replay_buffer.sample(batch_size)
        states = torch.from_numpy(np.asarray([s[0] for s in samples], dtype=np.float32))
        policies = torch.from_numpy(np.asarray([s[1] for s in samples], dtype=np.float32))
        values = torch.from_numpy(np.asarray([s[2] for s in samples], dtype=np.float32))
        if device.type == "cuda":
            states = states.pin_memory().to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            policies = policies.pin_memory().to(device, non_blocking=True)
            values = values.pin_memory().to(device, non_blocking=True)
        else:
            states, policies, values = states.to(device), policies.to(device), values.to(device)
        tl, pl, vl = train_one_batch(model, optimizer, states, policies, values, scaler=scaler)
        total_losses.append(tl); policy_losses.append(pl); value_losses.append(vl)

    return {"total_loss": total_losses, "policy_loss": policy_losses, "value_loss": value_losses}

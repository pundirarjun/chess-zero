"""Memory-efficient GPU training utilities.

The replay buffer stays in CPU RAM. Only one training batch is moved to CUDA
at a time.
"""

from __future__ import annotations

import numpy as np
import torch

from training.trainer import train_one_batch


def train_from_replay_buffer(
    model,
    optimizer,
    replay_buffer,
    batch_size=32,
    training_steps=10,
):
    """Train from a CPU replay buffer, transferring only one batch per step."""
    if batch_size <= 0 or training_steps <= 0:
        raise ValueError("batch_size and training_steps must be greater than zero")

    device = next(model.parameters()).device

    if len(replay_buffer) < batch_size:
        raise ValueError(
            f"Replay buffer contains {len(replay_buffer)} samples, "
            f"but batch size is {batch_size}."
        )

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    totals = [0.0, 0.0, 0.0]

    for step in range(1, training_steps + 1):

        # Sample only one small batch from CPU replay.
        samples = replay_buffer.sample(batch_size)

        # Pack only this batch.
        states = torch.from_numpy(
            np.asarray(
                [s[0] for s in samples],
                dtype=np.float32,
            )
        )

        policies = torch.from_numpy(
            np.asarray(
                [s[1] for s in samples],
                dtype=np.float32,
            )
        )

        values = torch.from_numpy(
            np.asarray(
                [s[2] for s in samples],
                dtype=np.float32,
            )
        )

        # Transfer ONLY this batch to GPU.
        if device.type == "cuda":
            states = states.pin_memory().to(
                device,
                non_blocking=True,
            )

            policies = policies.pin_memory().to(
                device,
                non_blocking=True,
            )

            values = values.pin_memory().to(
                device,
                non_blocking=True,
            )

            states = states.contiguous(
                memory_format=torch.channels_last
            )

        else:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

        # Train on this batch.
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

        if (
            step == 1
            or step % 10 == 0
            or step == training_steps
        ):
            print(
                f"Step {step}/{training_steps}: "
                f"Total={tl:.6f} | "
                f"Policy={pl:.6f} | "
                f"Value={vl:.6f}"
            )

        # Release this batch before sampling the next one.
        del states
        del policies
        del values
        del samples

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "average_total_loss": totals[0] / training_steps,
        "average_policy_loss": totals[1] / training_steps,
        "average_value_loss": totals[2] / training_steps,
    }


# ---------------------------------------------------------------------------
# Compatibility function
# ---------------------------------------------------------------------------

def train_from_gpu_replay(
    model,
    optimizer,
    gpu_replay,
    batch_size=128,
    training_steps=10,
):
    """Compatibility wrapper for callers that already have GPU replay.

    New training code should use train_from_replay_buffer().
    """

    states_all, policies_all, values_all = gpu_replay

    device = states_all.device

    if device.type != "cuda":
        raise ValueError("gpu_replay must be on CUDA")

    if batch_size <= 0 or training_steps <= 0:
        raise ValueError(
            "batch_size and training_steps must be greater than zero"
        )

    n = states_all.shape[0]

    if n < batch_size:
        raise ValueError(
            f"Replay buffer contains {n} samples, "
            f"but batch size is {batch_size}."
        )

    scaler = torch.amp.GradScaler("cuda")

    totals = [0.0, 0.0, 0.0]

    for step in range(1, training_steps + 1):

        indices = torch.randint(
            0,
            n,
            (batch_size,),
            device=device,
        )

        states = states_all.index_select(
            0,
            indices,
        )

        policies = policies_all.index_select(
            0,
            indices,
        )

        values = values_all.index_select(
            0,
            indices,
        )

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

        if (
            step == 1
            or step % 10 == 0
            or step == training_steps
        ):
            print(
                f"Step {step}/{training_steps}: "
                f"Total={tl:.6f} | "
                f"Policy={pl:.6f} | "
                f"Value={vl:.6f}"
            )

    return {
        "average_total_loss": totals[0] / training_steps,
        "average_policy_loss": totals[1] / training_steps,
        "average_value_loss": totals[2] / training_steps,
    }
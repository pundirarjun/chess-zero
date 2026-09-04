import torch

from training.dataset import ChessDataset
from training.trainer import train_one_batch


def train_from_replay_buffer(
    model,
    optimizer,
    replay_buffer,
    batch_size=32,
    training_steps=10
):

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

    if training_steps <= 0:
        raise ValueError(
            "training_steps must be greater than 0."
        )

    if len(replay_buffer) < batch_size:
        raise ValueError(
            f"Replay buffer contains "
            f"{len(replay_buffer)} samples, "
            f"but batch size is {batch_size}."
        )

    total_losses = []
    policy_losses = []
    value_losses = []

    for step in range(training_steps):

        samples = replay_buffer.sample(
            batch_size
        )

        dataset = ChessDataset(
            samples
        )

        states = []
        policies = []
        values = []

        for i in range(len(dataset)):

            state, policy, value = dataset[i]

            states.append(state)
            policies.append(policy)
            values.append(value)

        states = torch.stack(
            states
        )

        policies = torch.stack(
            policies
        )

        values = torch.stack(
            values
        )

        total_loss, policy_loss, value_loss = (
            train_one_batch(
                model=model,
                optimizer=optimizer,
                states=states,
                target_policy=policies,
                target_value=values
            )
        )

        total_losses.append(
            total_loss
        )

        policy_losses.append(
            policy_loss
        )

        value_losses.append(
            value_loss
        )

    return {
        "total_loss": total_losses,
        "policy_loss": policy_losses,
        "value_loss": value_losses
    }
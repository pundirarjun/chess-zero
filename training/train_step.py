import torch

from training.dataset import ChessDataset
from training.loss import alpha_zero_loss


def train_from_replay_buffer(
    model,
    optimizer,
    replay_buffer,
    batch_size=32,
    training_steps=10
):

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

        states = torch.stack(states)
        policies = torch.stack(policies)
        values = torch.stack(values)

        model.train()

        optimizer.zero_grad()

        predicted_policy, predicted_value = model(
            states
        )

        total_loss, policy_loss, value_loss = alpha_zero_loss(
            predicted_policy,
            predicted_value,
            policies,
            values
        )

        total_loss.backward()

        optimizer.step()

        total_losses.append(
            total_loss.item()
        )

        policy_losses.append(
            policy_loss.item()
        )

        value_losses.append(
            value_loss.item()
        )

    return {
        "total_loss": total_losses,
        "policy_loss": policy_losses,
        "value_loss": value_losses
    }
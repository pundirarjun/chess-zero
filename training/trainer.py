import torch

from training.loss import alpha_zero_loss


def train_one_batch(
    model,
    optimizer,
    states,
    target_policy,
    target_value
):

    model.train()

    optimizer.zero_grad()

    predicted_policy, predicted_value = model(
        states
    )

    total_loss, policy_loss, value_loss = alpha_zero_loss(
        predicted_policy,
        predicted_value,
        target_policy,
        target_value
    )

    total_loss.backward()

    optimizer.step()

    return (
        total_loss.item(),
        policy_loss.item(),
        value_loss.item()
    )
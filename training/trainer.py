import torch

from training.loss import alpha_zero_loss


def train_one_batch(
    model,
    optimizer,
    states,
    target_policy,
    target_value,
    scaler=None
):

    model.train()

    optimizer.zero_grad(set_to_none=True)

    device_type = (
        states.device.type
        if states.is_cuda
        else "cpu"
    )

    use_amp = (
        device_type == "cuda"
    )

    if use_amp:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            predicted_policy, predicted_value = model(
                states
            )

            total_loss, policy_loss, value_loss = (
                alpha_zero_loss(
                    predicted_policy,
                    predicted_value,
                    target_policy,
                    target_value
                )
            )
    else:
        predicted_policy, predicted_value = model(
            states
        )

        total_loss, policy_loss, value_loss = (
            alpha_zero_loss(
                predicted_policy,
                predicted_value,
                target_policy,
                target_value
            )
        )

    if scaler is not None and use_amp:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        optimizer.step()

    return (
        total_loss.item(),
        policy_loss.item(),
        value_loss.item()
    )

import torch
import torch.nn.functional as F


def alpha_zero_loss(
    policy_logits,
    value_pred,
    target_policy,
    target_value
):

    # Policy loss

    log_probs = F.log_softmax(
        policy_logits,
        dim=1
    )

    policy_loss = -(
        target_policy * log_probs
    ).sum(dim=1).mean()


    # Value loss

    target_value = target_value.unsqueeze(1)

    value_loss = F.mse_loss(
        value_pred,
        target_value
    )


    # Total loss

    total_loss = (
        policy_loss +
        value_loss
    )

    return (
        total_loss,
        policy_loss,
        value_loss
    )
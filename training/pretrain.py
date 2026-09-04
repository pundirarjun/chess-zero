import torch
import torch.nn.functional as F


def pretrain_one_epoch(
    model,
    dataloader,
    optimizer,
    device="cpu"
):

    model.train()

    total_loss = 0.0

    total_policy_loss = 0.0

    total_value_loss = 0.0

    batches = 0

    # ======================================================
    # TRAINING LOOP
    # ======================================================

    for states, policies, values in dataloader:

        # --------------------------------------------------
        # Move data to device
        # --------------------------------------------------

        states = states.to(
            device,
            non_blocking=True
        )

        policies = policies.to(
            device,
            non_blocking=True
        )

        values = values.to(
            device,
            non_blocking=True
        )

        # --------------------------------------------------
        # Reset gradients
        # --------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        policy_logits, value_pred = model(
            states
        )

        # ==================================================
        # POLICY LOSS
        # ==================================================

        # PGN policy targets are one-hot.

        target_actions = policies.argmax(
            dim=1
        )

        policy_loss = F.cross_entropy(
            policy_logits,
            target_actions
        )

        # ==================================================
        # VALUE LOSS
        # ==================================================

        value_pred = value_pred.squeeze(
            1
        )

        value_loss = F.mse_loss(
            value_pred,
            values
        )

        # ==================================================
        # TOTAL LOSS
        # ==================================================

        loss = (
            policy_loss
            +
            value_loss
        )

        # ==================================================
        # BACKPROPAGATION
        # ==================================================

        loss.backward()

        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # --------------------------------------------------
        # Optimizer step
        # --------------------------------------------------

        optimizer.step()

        # ==================================================
        # STATISTICS
        # ==================================================

        total_loss += (
            loss.item()
        )

        total_policy_loss += (
            policy_loss.item()
        )

        total_value_loss += (
            value_loss.item()
        )

        batches += 1

    # ======================================================
    # EMPTY DATALOADER CHECK
    # ======================================================

    if batches == 0:

        raise RuntimeError(
            "Dataloader produced zero batches."
        )

    # ======================================================
    # RETURN AVERAGES
    # ======================================================

    return {

        "total_loss":
            total_loss / batches,

        "policy_loss":
            total_policy_loss / batches,

        "value_loss":
            total_value_loss / batches
    }
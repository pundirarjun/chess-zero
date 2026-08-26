import torch


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


    for states, policies, values in dataloader:

        states = states.to(
            device
        )

        policies = policies.to(
            device
        )

        values = values.to(
            device
        )


        optimizer.zero_grad()


        policy_logits, value_pred = model(
            states
        )


        # ------------------------------------------
        # Policy loss
        # ------------------------------------------

        target_actions = policies.argmax(
            dim=1
        )


        policy_loss = torch.nn.functional.cross_entropy(

            policy_logits,

            target_actions
        )


        # ------------------------------------------
        # Value loss
        # ------------------------------------------

        value_pred = value_pred.squeeze(
            1
        )


        value_loss = torch.nn.functional.mse_loss(

            value_pred,

            values
        )


        # ------------------------------------------
        # Combined loss
        # ------------------------------------------

        loss = (
            policy_loss
            +
            value_loss
        )


        loss.backward()


        optimizer.step()


        total_loss += loss.item()

        total_policy_loss += (
            policy_loss.item()
        )

        total_value_loss += (
            value_loss.item()
        )

        batches += 1


    return {

        "total_loss":
            total_loss / batches,

        "policy_loss":
            total_policy_loss / batches,

        "value_loss":
            total_value_loss / batches
    }
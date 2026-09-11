"""Single-batch mixed-precision trainer."""

from __future__ import annotations

import torch

from training.loss import alpha_zero_loss


def train_one_batch(model, optimizer, states, target_policy, target_value, scaler=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    use_amp = states.device.type == "cuda"
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predicted_policy, predicted_value = model(states)
            total_loss, policy_loss, value_loss = alpha_zero_loss(
                predicted_policy, predicted_value, target_policy, target_value
            )
    else:
        predicted_policy, predicted_value = model(states)
        total_loss, policy_loss, value_loss = alpha_zero_loss(
            predicted_policy, predicted_value, target_policy, target_value
        )
    if use_amp and scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return total_loss.item(), policy_loss.item(), value_loss.item()

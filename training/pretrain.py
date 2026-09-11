"""GPU-first supervised pretraining utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pretrain_one_epoch(model, dataloader, optimizer, device="cpu"):
    device = torch.device(device)
    model.train()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    totals = [0.0, 0.0, 0.0]
    batches = 0

    for states, actions, values in dataloader:
        states = states.to(device, non_blocking=device.type == "cuda")
        actions = actions.to(device, non_blocking=device.type == "cuda")
        values = values.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            states = states.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, pred_value = model(states)
                policy_loss = F.cross_entropy(logits, actions)
                value_loss = F.mse_loss(pred_value.squeeze(-1), values)
                loss = policy_loss + value_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, pred_value = model(states)
            policy_loss = F.cross_entropy(logits, actions)
            value_loss = F.mse_loss(pred_value.squeeze(-1), values)
            loss = policy_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        totals[0] += float(loss.item()); totals[1] += float(policy_loss.item()); totals[2] += float(value_loss.item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Dataloader produced zero batches.")
    return {"total_loss": totals[0] / batches, "policy_loss": totals[1] / batches, "value_loss": totals[2] / batches}

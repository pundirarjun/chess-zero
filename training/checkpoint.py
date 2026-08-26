import torch
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    iteration,
    path
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    model,
    optimizer,
    path
):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return checkpoint["iteration"]
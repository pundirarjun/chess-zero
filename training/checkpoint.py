import torch

from pathlib import Path


# ==========================================================
# SAVE CHECKPOINT
# ==========================================================

def save_checkpoint(
    model,
    optimizer,
    iteration,
    path,
    **extra
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "iteration":
            iteration,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict()

    }

    # ------------------------------------------------------
    # Optional extra information
    # ------------------------------------------------------

    checkpoint.update(
        extra
    )

    torch.save(
        checkpoint,
        path
    )


# ==========================================================
# LOAD CHECKPOINT
# ==========================================================

def load_checkpoint(
    model,
    optimizer,
    path,
    device=None
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    # ------------------------------------------------------
    # Determine device
    # ------------------------------------------------------

    if device is None:

        device = next(
            model.parameters()
        ).device

    # ------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ------------------------------------------------------
    # Load optimizer
    # ------------------------------------------------------

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    # ------------------------------------------------------
    # Move optimizer state to model device
    # ------------------------------------------------------

    for state in optimizer.state.values():

        for key, value in state.items():

            if torch.is_tensor(value):

                state[key] = value.to(
                    device
                )

    # ------------------------------------------------------
    # Return complete checkpoint
    # ------------------------------------------------------

    return checkpoint
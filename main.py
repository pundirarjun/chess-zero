import sys
import os
import random

# ==========================================================
# MAKE PROJECT ROOT IMPORTABLE
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# IMPORTS
# ==========================================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from model.chess_net import ChessNet
from environment.action_encoder import ActionEncoder

from training.pgn_dataset import PGNDatasetBuilder
from training.dataset import ChessDataset
from training.checkpoint import save_checkpoint


# ==========================================================
# CONFIGURATION
# ==========================================================

# ----------------------------------------------------------
# PGN DATASET
# ----------------------------------------------------------

PGN_PATH = (
    "/kaggle/input/datasets/"
    "arjunthakur9999/chess-dataset/"
    "lichess_2013_01.pgn"
)

NUM_GAMES = 10000

# ----------------------------------------------------------
# TRAINING
# ----------------------------------------------------------

EPOCHS = 3

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

VALIDATION_SPLIT = 0.05

RANDOM_SEED = 42


# ----------------------------------------------------------
# CHECKPOINT
# ----------------------------------------------------------

CHECKPOINT_PATH = (
    "checkpoints/pretrained_phase1.pt"
)


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    DEVICE
)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ==========================================================
# ACTION ENCODER
# ==========================================================

action_encoder = ActionEncoder()

ACTION_SPACE_SIZE = (
    action_encoder.size()
)

print(
    "Action space:",
    ACTION_SPACE_SIZE
)


# ==========================================================
# BUILD PGN DATASET
# ==========================================================

print(
    "\n=============================="
)

print(
    "BUILDING PGN DATASET"
)

print(
    "=============================="
)

print(
    "PGN:",
    PGN_PATH
)

print(
    "Games:",
    NUM_GAMES
)


builder = PGNDatasetBuilder()

samples = builder.build_from_pgn(
    PGN_PATH,
    max_games=NUM_GAMES
)

if len(samples) == 0:
    raise RuntimeError(
        "No training samples were created."
    )


print(
    "\nTotal samples:",
    len(samples)
)


# ==========================================================
# TRAIN / VALIDATION SPLIT
# ==========================================================

dataset = ChessDataset(
    samples
)

validation_size = max(
    1,
    int(
        len(dataset)
        * VALIDATION_SPLIT
    )
)

training_size = (
    len(dataset)
    - validation_size
)

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset = (
    random_split(
        dataset,
        [
            training_size,
            validation_size
        ],
        generator=generator
    )
)


print(
    "Training samples:",
    training_size
)

print(
    "Validation samples:",
    validation_size
)


# ==========================================================
# DATA LOADERS
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


# ==========================================================
# CREATE MODEL
# ==========================================================

print(
    "\n=============================="
)

print(
    "CREATING MODEL"
)

print(
    "=============================="
)


model = ChessNet(
    action_space_size=ACTION_SPACE_SIZE
)

model.to(
    DEVICE
)


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ==========================================================
# TRAINING FUNCTION
# ==========================================================

def train_epoch():

    model.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0

    total_samples = 0

    for states, policies, values in train_loader:

        states = states.to(
            DEVICE,
            non_blocking=True
        )

        policies = policies.to(
            DEVICE,
            non_blocking=True
        )

        values = values.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        policy_logits, value_pred = (
            model(states)
        )

        # --------------------------------------------------
        # Policy loss
        # --------------------------------------------------

        target_actions = (
            policies.argmax(
                dim=1
            )
        )

        policy_loss = (
            F.cross_entropy(
                policy_logits,
                target_actions
            )
        )

        # --------------------------------------------------
        # Value loss
        # --------------------------------------------------

        value_pred = (
            value_pred.squeeze(-1)
        )

        value_loss = F.mse_loss(
            value_pred,
            values
        )

        # --------------------------------------------------
        # Total loss
        # --------------------------------------------------

        loss = (
            policy_loss
            + value_loss
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            states.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_policy_loss += (
            policy_loss.item()
            * batch_size
        )

        total_value_loss += (
            value_loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss / total_samples,
        total_policy_loss / total_samples,
        total_value_loss / total_samples
    )


# ==========================================================
# VALIDATION FUNCTION
# ==========================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0

    total_samples = 0

    for states, policies, values in validation_loader:

        states = states.to(
            DEVICE,
            non_blocking=True
        )

        policies = policies.to(
            DEVICE,
            non_blocking=True
        )

        values = values.to(
            DEVICE,
            non_blocking=True
        )

        policy_logits, value_pred = (
            model(states)
        )

        target_actions = (
            policies.argmax(
                dim=1
            )
        )

        policy_loss = (
            F.cross_entropy(
                policy_logits,
                target_actions
            )
        )

        value_pred = (
            value_pred.squeeze(-1)
        )

        value_loss = F.mse_loss(
            value_pred,
            values
        )

        loss = (
            policy_loss
            + value_loss
        )

        batch_size = (
            states.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_policy_loss += (
            policy_loss.item()
            * batch_size
        )

        total_value_loss += (
            value_loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss / total_samples,
        total_policy_loss / total_samples,
        total_value_loss / total_samples
    )


# ==========================================================
# PRETRAINING
# ==========================================================

print(
    "\n=============================="
)

print(
    "PHASE 1 PRETRAINING"
)

print(
    "=============================="
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)


for epoch in range(
    1,
    EPOCHS + 1
):

    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    train_loss, train_policy, train_value = (
        train_epoch()
    )

    validation_loss, validation_policy, validation_value = (
        validate()
    )

    print(
        f"Train total: {train_loss:.6f}"
    )

    print(
        f"Train policy: {train_policy:.6f}"
    )

    print(
        f"Train value: {train_value:.6f}"
    )

    print(
        f"Validation total: "
        f"{validation_loss:.6f}"
    )

    print(
        f"Validation policy: "
        f"{validation_policy:.6f}"
    )

    print(
        f"Validation value: "
        f"{validation_value:.6f}"
    )

    # ------------------------------------------------------
    # Save checkpoint after every epoch
    # ------------------------------------------------------

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=epoch,
        path=CHECKPOINT_PATH,
        epoch=epoch,
        num_games=NUM_GAMES,
        num_samples=len(samples),
        train_loss=train_loss,
        train_policy_loss=train_policy,
        train_value_loss=train_value,
        validation_loss=validation_loss,
        validation_policy_loss=validation_policy,
        validation_value_loss=validation_value
    )

    print(
        "Checkpoint saved:",
        CHECKPOINT_PATH
    )


# ==========================================================
# COMPLETE
# ==========================================================

print(
    "\n=============================="
)

print(
    "PHASE 1 PRETRAINING COMPLETE"
)

print(
    "=============================="
)

print(
    "Games:",
    NUM_GAMES
)

print(
    "Samples:",
    len(samples)
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

print(
    "\nThis checkpoint can now be"
)

print(
    "used as the starting model"
)

print(
    "for your existing RL pipeline."
)
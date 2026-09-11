"""Phase-1 supervised pretraining.

PGN parsing/validation is CPU-side because the source is a PGN text format.
Once samples exist, model forward/backward, loss, and optimizer work run on CUDA
with pinned host batches and FP16 autocast.
"""

from __future__ import annotations

import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from model.chess_net import ChessNet
from environment.action_encoder import ActionEncoder
from training.pgn_dataset import PGNDatasetBuilder
from training.checkpoint import save_checkpoint

PGN_PATH = "/kaggle/input/datasets/arjunthakur9999/chess-dataset/lichess_2013_01.pgn"
NUM_GAMES = 10000
EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT = 0.05
RANDOM_SEED = 42
CHECKPOINT_PATH = "checkpoints/pretrained_phase1.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


class CompactPGNDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        state, action_id, value = self.samples[index]
        return (
            torch.from_numpy(state),
            torch.tensor(action_id, dtype=torch.long),
            torch.tensor(value, dtype=torch.float32),
        )


def run():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    encoder = ActionEncoder()
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
        print("FP16 AMP: enabled")
    print("Action space:", encoder.size())

    print("\n==============================")
    print("BUILDING PGN DATASET")
    print("==============================")
    print("PGN:", PGN_PATH)
    print("Games:", NUM_GAMES)
    builder = PGNDatasetBuilder()
    samples = builder.build_from_pgn(PGN_PATH, max_games=NUM_GAMES)
    if not samples:
        raise RuntimeError("No training samples were created.")

    dataset = CompactPGNDataset(samples)
    validation_size = max(1, int(len(dataset) * VALIDATION_SPLIT))
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, validation_dataset = random_split(dataset, [training_size, validation_size], generator=generator)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    model = ChessNet(action_space_size=encoder.size()).to(DEVICE)
    if DEVICE.type == "cuda":
        model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    def train_epoch():
        model.train()
        totals = [0.0, 0.0, 0.0, 0]
        for states, actions, values in train_loader:
            states = states.to(DEVICE, non_blocking=pin)
            actions = actions.to(DEVICE, non_blocking=pin)
            values = values.to(DEVICE, non_blocking=pin)
            if DEVICE.type == "cuda":
                states = states.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            if DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, pred = model(states)
                    pl = F.cross_entropy(logits, actions)
                    vl = F.mse_loss(pred.squeeze(-1), values)
                    loss = pl + vl
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, pred = model(states)
                pl = F.cross_entropy(logits, actions)
                vl = F.mse_loss(pred.squeeze(-1), values)
                loss = pl + vl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n = states.size(0)
            totals[0] += loss.item() * n; totals[1] += pl.item() * n; totals[2] += vl.item() * n; totals[3] += n
        return tuple(x / totals[3] for x in totals[:3])

    @torch.no_grad()
    def validate():
        model.eval()
        totals = [0.0, 0.0, 0.0, 0]
        for states, actions, values in validation_loader:
            states = states.to(DEVICE, non_blocking=pin)
            actions = actions.to(DEVICE, non_blocking=pin)
            values = values.to(DEVICE, non_blocking=pin)
            if DEVICE.type == "cuda":
                states = states.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, pred = model(states)
                    pl = F.cross_entropy(logits, actions); vl = F.mse_loss(pred.squeeze(-1), values); loss = pl + vl
            else:
                logits, pred = model(states); pl = F.cross_entropy(logits, actions); vl = F.mse_loss(pred.squeeze(-1), values); loss = pl + vl
            n = states.size(0)
            totals[0] += loss.item() * n; totals[1] += pl.item() * n; totals[2] += vl.item() * n; totals[3] += n
        return tuple(x / totals[3] for x in totals[:3])

    print("\n==============================")
    print("PHASE 1 PRETRAINING")
    print("==============================")
    for epoch in range(1, EPOCHS + 1):
        tr = train_epoch(); va = validate()
        print(f"Epoch {epoch}/{EPOCHS} | Train={tr[0]:.6f} (P={tr[1]:.6f}, V={tr[2]:.6f}) | Val={va[0]:.6f} (P={va[1]:.6f}, V={va[2]:.6f})")
        save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)
    print("\nPhase 1 complete:", CHECKPOINT_PATH)


if __name__ == "__main__":
    run()

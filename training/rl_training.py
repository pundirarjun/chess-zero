"""GPU-first reinforcement-learning training for the chess AI.

Usage on Kaggle:
    python training/rl_training.py

Set RL_ITERATION to the next number in the chain.  The script loads the prior
iteration, runs GPU-native self-play/MCTS, appends completed games to the replay
buffer, and trains the unchanged 18x8x8 -> 4544 policy/value network on CUDA.
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.chess_net import ChessNet
from environment.action_encoder import ActionEncoder
from training.self_play import play_games
from training.replay_buffer import ReplayBuffer
from training.train_step import build_gpu_replay, train_from_gpu_replay


# ---------------------------------------------------------------------------
# Iteration / paths
# ---------------------------------------------------------------------------
RL_ITERATION = 22
PREVIOUS_ITERATION = RL_ITERATION - 1
KAGGLE_CHECKPOINT_ROOT = "/kaggle/working/chess-zero/checkpoints"
LOCAL_CHECKPOINT_ROOT = "checkpoints"   
PREVIOUS_CHECKPOINT = os.path.join(KAGGLE_CHECKPOINT_ROOT, f"rl_iteration_{PREVIOUS_ITERATION}.pt")
PREVIOUS_REPLAY_BUFFER = os.path.join(KAGGLE_CHECKPOINT_ROOT, f"replay_buffer_rl{PREVIOUS_ITERATION}.pt")
OUTPUT_CHECKPOINT = os.path.join(LOCAL_CHECKPOINT_ROOT, f"rl_iteration_{RL_ITERATION}.pt")
OUTPUT_REPLAY_BUFFER = os.path.join(LOCAL_CHECKPOINT_ROOT, f"replay_buffer_rl{RL_ITERATION}.pt")


# ---------------------------------------------------------------------------
# Self-play.  Increase games first to keep the GPU busy; then increase sims.
# ---------------------------------------------------------------------------
NUM_SELF_PLAY_GAMES = 256
NUM_SIMULATIONS = 100
MAX_MOVES = 400
MCTS_BATCH_SIZE = 64  # GPU search batch width / active-game batch.
TEMPERATURE = 1.0
TEMPERATURE_MOVES = 60
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25


# ---------------------------------------------------------------------------
# Replay / training
# ---------------------------------------------------------------------------
REPLAY_BUFFER_CAPACITY = 200000
TRAINING_BATCH_SIZE = 256
TRAINING_STEPS = 150
LEARNING_RATE = 1e-4
SEED = 42


def configure_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


device = configure_device()


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------
def create_model():
    encoder = ActionEncoder()
    model = ChessNet(action_space_size=encoder.size()).to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    return model, encoder


def create_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------------------
# Checkpoint / replay loading
# ---------------------------------------------------------------------------
def _resolve_path(path):
    if os.path.exists(path):
        return path
    local = os.path.join(LOCAL_CHECKPOINT_ROOT, os.path.basename(path))
    if os.path.exists(local):
        return local
    raise FileNotFoundError(f"Required file not found: {path}")


def load_previous_checkpoint(model, optimizer, path):
    path = _resolve_path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Restore both model weights and Adam state so each RL iteration continues
    # from the previous optimizer state instead of restarting Adam from scratch.
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print("Loaded previous optimizer state.")
    else:
        # Keep compatibility with older checkpoints that did not save it.
        print("Warning: previous checkpoint has no optimizer state; using a fresh Adam optimizer.")

    # The learning rate is intentionally controlled by this iteration's config,
    # even when the previous checkpoint was trained with a different LR.
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE

    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    model.eval()
    return checkpoint, path


def load_previous_replay_buffer(path):
    path = _resolve_path(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, list):
        raise RuntimeError("Invalid replay buffer format: expected a list.")
    replay = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
    replay.add(data)
    return replay, path


def generate_self_play_data(model, replay_buffer):
    print("\n==============================")
    print("GPU SELF-PLAY")
    print("==============================")
    print(f"Games: {NUM_SELF_PLAY_GAMES}")
    print(f"Simulations/game: {NUM_SIMULATIONS}")
    print(f"Max moves: {MAX_MOVES}")

    results = play_games(
        model=model,
        num_games=NUM_SELF_PLAY_GAMES,
        num_simulations=NUM_SIMULATIONS,
        max_moves=MAX_MOVES,
        temperature=TEMPERATURE,
        temperature_moves=TEMPERATURE_MOVES,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        batch_size=MCTS_BATCH_SIZE,
    )

    completed = incomplete = white_wins = black_wins = draws = new_samples = 0
    termination_counts = {}
    for result in results:
        termination_counts[result.termination] = termination_counts.get(result.termination, 0) + 1
        if not result.completed:
            incomplete += 1
            continue
        completed += 1
        replay_buffer.add(result.training_data)
        new_samples += len(result.training_data)
        if result.result == 1:
            white_wins += 1
        elif result.result == -1:
            black_wins += 1
        else:
            draws += 1

    stats = {
        "completed_games": completed,
        "incomplete_games": incomplete,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "new_samples": new_samples,
        "termination_counts": termination_counts,
        "replay_buffer_size": len(replay_buffer),
    }
    print("\nSelf-play summary:", stats)
    return stats


def save_replay_buffer(replay_buffer, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(list(replay_buffer), path)
    print("Replay buffer saved:", path)


def train_model(model, optimizer, replay_buffer):
    if len(replay_buffer) < TRAINING_BATCH_SIZE:
        raise RuntimeError(f"Replay buffer has {len(replay_buffer)} samples; need {TRAINING_BATCH_SIZE}.")

    print("\n==============================")
    print("GPU TRAINING")
    print("==============================")
    print("Loading replay buffer to GPU once...")

    # The complete replay dataset is transferred to CUDA once. Each subsequent
    # batch is sampled with GPU-generated indices, eliminating the old
    # Python-list -> NumPy -> pinned-memory -> CUDA pipeline on every step.
    gpu_replay = build_gpu_replay(replay_buffer, device)
    print(f"GPU replay samples: {gpu_replay[0].shape[0]}")

    return train_from_gpu_replay(
        model=model,
        optimizer=optimizer,
        gpu_replay=gpu_replay,
        batch_size=TRAINING_BATCH_SIZE,
        training_steps=TRAINING_STEPS,
    )


def save_rl_checkpoint(model, optimizer, self_play_stats, training_stats):
    os.makedirs(os.path.dirname(OUTPUT_CHECKPOINT) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": RL_ITERATION,
        "previous_checkpoint": PREVIOUS_CHECKPOINT,
        **self_play_stats,
        "num_simulations": NUM_SIMULATIONS,
        "mcts_batch_size": MCTS_BATCH_SIZE,
        "temperature": TEMPERATURE,
        "temperature_moves": TEMPERATURE_MOVES,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "dirichlet_epsilon": DIRICHLET_EPSILON,
        "max_moves": MAX_MOVES,
        "training_batch_size": TRAINING_BATCH_SIZE,
        "training_steps": TRAINING_STEPS,
        "learning_rate": LEARNING_RATE,
        **training_stats,
    }
    torch.save(checkpoint, OUTPUT_CHECKPOINT)
    print("Model checkpoint saved:", OUTPUT_CHECKPOINT)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("==========================================")
    print("GPU RL TRAINING")
    print("==========================================")
    print("Iteration:", RL_ITERATION)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
        print("FP16 AMP: enabled")
        print("Channels last: enabled")
        print("TF32: enabled")
    print("Previous checkpoint:", PREVIOUS_CHECKPOINT)
    print("Previous replay buffer:", PREVIOUS_REPLAY_BUFFER)
    print("Output checkpoint:", OUTPUT_CHECKPOINT)
    print("Output replay buffer:", OUTPUT_REPLAY_BUFFER)

    model, encoder = create_model()
    optimizer = create_optimizer(model)
    print("Action space size:", encoder.size())

    checkpoint, resolved_checkpoint = load_previous_checkpoint(model, optimizer, PREVIOUS_CHECKPOINT)
    print("Loaded previous checkpoint:", resolved_checkpoint)
    print("Previous iteration:", checkpoint.get("iteration"))

    replay, resolved_replay = load_previous_replay_buffer(PREVIOUS_REPLAY_BUFFER)
    print("Loaded replay buffer:", resolved_replay)
    print("Previous samples:", len(replay))

    self_play_stats = generate_self_play_data(model, replay)
    save_replay_buffer(replay, OUTPUT_REPLAY_BUFFER)
    training_stats = train_model(model, optimizer, replay)
    save_rl_checkpoint(model, optimizer, self_play_stats, training_stats)

    print("\n==========================================")
    print(f"RL ITERATION {RL_ITERATION} COMPLETE")
    print("==========================================")


if __name__ == "__main__":
    main()

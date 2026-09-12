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
from training.trainer import train_one_batch


# ---------------------------------------------------------------------------
# Iteration / paths
# ---------------------------------------------------------------------------
RL_ITERATION = 10
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
NUM_SELF_PLAY_GAMES = 64
NUM_SIMULATIONS = 100
MAX_MOVES = 400
MCTS_BATCH_SIZE = 32  # API compatibility; GPU MCTS batches all active games.
TEMPERATURE = 1.0
TEMPERATURE_MOVES = 60
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25


# ---------------------------------------------------------------------------
# Replay / training
# ---------------------------------------------------------------------------
REPLAY_BUFFER_CAPACITY = 50000
TRAINING_BATCH_SIZE = 128
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


def create_scaler():
    if device.type != "cuda":
        return None
    return torch.amp.GradScaler("cuda")


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
    model.load_state_dict(checkpoint["model_state_dict"])
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

    scaler = create_scaler()
    totals = [0.0, 0.0, 0.0]
    print("\n==============================")
    print("GPU TRAINING")
    print("==============================")

    for step in range(1, TRAINING_STEPS + 1):
        batch = replay_buffer.sample(TRAINING_BATCH_SIZE)
        states_np = np.asarray([x[0] for x in batch], dtype=np.float32)
        policies_np = np.asarray([x[1] for x in batch], dtype=np.float32)
        values_np = np.asarray([x[2] for x in batch], dtype=np.float32)

        states = torch.from_numpy(states_np)
        policies = torch.from_numpy(policies_np)
        values = torch.from_numpy(values_np)
        if device.type == "cuda":
            states = states.pin_memory().to(device, non_blocking=True)
            policies = policies.pin_memory().to(device, non_blocking=True)
            values = values.pin_memory().to(device, non_blocking=True)
            states = states.contiguous(memory_format=torch.channels_last)
        else:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

        loss, policy_loss, value_loss = train_one_batch(
            model=model,
            optimizer=optimizer,
            states=states,
            target_policy=policies,
            target_value=values,
            scaler=scaler,
        )
        totals[0] += loss; totals[1] += policy_loss; totals[2] += value_loss
        if step == 1 or step % 10 == 0 or step == TRAINING_STEPS:
            print(f"Step {step}/{TRAINING_STEPS}: Total={loss:.6f} | Policy={policy_loss:.6f} | Value={value_loss:.6f}")

    return {
        "average_total_loss": totals[0] / TRAINING_STEPS,
        "average_policy_loss": totals[1] / TRAINING_STEPS,
        "average_value_loss": totals[2] / TRAINING_STEPS,
    }


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

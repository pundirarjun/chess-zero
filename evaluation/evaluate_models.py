"""GPU-native model evaluation.

This replaces per-move python-chess/Node MCTS with the tensorized GPU chess
engine.  Checkpoint loading remains standard PyTorch file I/O on the CPU, while
search, state transitions, legal moves, and neural inference run on CUDA.
"""

from __future__ import annotations

import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


NUM_GAMES = 10
NUM_SIMULATIONS = 50
MAX_MOVES = 300
EVALUATION_TEMPERATURE = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_CHECKPOINT = "/kaggle/input/datasets/arjunthakur9999/checkpoints/pretrained_phase1.pt"
RL_CHECKPOINT = "/kaggle/input/datasets/arjunthakur9999/checkpoints/rl_iteration_1.pt"

ACTION_SPACE_SIZE = 4544


def create_model(path):
    model = ChessNet(action_space_size=ACTION_SPACE_SIZE).to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if DEVICE.type == "cuda":
        model.to(memory_format=torch.channels_last)
    return model, checkpoint


def terminal_result(state: GPUChess):
    terminal, _ = state.terminal_info()
    if not bool(terminal[0].item()):
        return None
    if bool(state.halfmove_clock[0].item() >= 100):
        return "1/2-1/2", "FIFTY_MOVE"
    if bool(state.insufficient_material()[0].item()):
        return "1/2-1/2", "INSUFFICIENT_MATERIAL"
    if not bool(state.legal_move_mask()[0].any().item()):
        if bool(state.is_in_check()[0].item()):
            # side to move is checkmated
            return ("0-1" if not bool(state.turn[0].item()) else "1-0"), "CHECKMATE"
        return "1/2-1/2", "STALEMATE"
    return None


def get_move(model, state):
    search = GPUMCTS(model=model, device=DEVICE)
    search.search(state, NUM_SIMULATIONS)
    actions = search.select_actions(EVALUATION_TEMPERATURE)
    return actions, search.advance(actions)


def play_game(white_model, black_model):
    state = GPUChess(DEVICE, 1)
    repetition = {}
    moves = 0
    while moves < MAX_MOVES:
        key = int(state.state_hash()[0].item())
        repetition[key] = repetition.get(key, 0) + 1
        if repetition[key] >= 3:
            return {"result": "1/2-1/2", "termination": "THREEFOLD_REPETITION", "moves": moves}

        done = terminal_result(state)
        if done is not None:
            return {"result": done[0], "termination": done[1], "moves": moves}

        model = white_model if not bool(state.turn[0].item()) else black_model
        _, state = get_move(model, state)
        moves += 1

    return {"result": None, "termination": "MAX_MOVES", "moves": moves}


def evaluate_models(model_a, model_b, name_a, name_b):
    a_wins = b_wins = draws = truncated = total_moves = 0
    terminations = {}
    for game in range(1, NUM_GAMES + 1):
        if game % 2:
            white, black, a_color = model_a, model_b, "White"
        else:
            white, black, a_color = model_b, model_a, "Black"
        result = play_game(white, black)
        print(f"Game {game}/{NUM_GAMES}: {name_a}={a_color} | {result}")
        total_moves += result["moves"]
        terminations[result["termination"]] = terminations.get(result["termination"], 0) + 1
        if result["result"] is None:
            truncated += 1
        elif result["result"] == "1/2-1/2":
            draws += 1
        elif (result["result"] == "1-0" and a_color == "White") or (result["result"] == "0-1" and a_color == "Black"):
            a_wins += 1
        else:
            b_wins += 1
    completed = a_wins + b_wins + draws
    a_score = (a_wins + 0.5 * draws) / completed if completed else 0.0
    b_score = (b_wins + 0.5 * draws) / completed if completed else 0.0
    print("\n==============================")
    print("GPU EVALUATION RESULTS")
    print("==============================")
    print(f"{name_a} wins:", a_wins)
    print(f"{name_b} wins:", b_wins)
    print("Draws:", draws)
    print("Truncated:", truncated)
    print("Average moves:", total_moves / NUM_GAMES if NUM_GAMES else 0.0)
    print("Scores:", name_a, a_score, "|", name_b, b_score)
    print("Terminations:", terminations)
    return {"a_wins": a_wins, "b_wins": b_wins, "draws": draws, "truncated": truncated, "a_score": a_score, "b_score": b_score, "termination_counts": terminations}


if __name__ == "__main__":
    if DEVICE.type != "cuda":
        raise RuntimeError("GPU evaluation requires CUDA.")
    print("Evaluation device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(0))
    pretrained, _ = create_model(PRETRAINED_CHECKPOINT)
    rl, _ = create_model(RL_CHECKPOINT)
    evaluate_models(pretrained, rl, "Phase 1 Pretrained", "RL Iteration 1")

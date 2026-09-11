"""GPU-native head-to-head evaluation helper."""

from __future__ import annotations

import torch

from environment.gpu_chess import GPUChess, _action_tables
from mcts.gpu_mcts import GPUMCTS


_FILES = "abcdefgh"
_RANKS = "12345678"
_PROMO = {2: "n", 3: "b", 4: "r", 5: "q"}


def action_to_uci(action: int) -> str:
    t = _action_tables()
    f = int(t["from_sq"][action]); to = int(t["to_sq"][action]); p = int(t["promo"][action])
    uci = _FILES[f % 8] + _RANKS[f // 8] + _FILES[to % 8] + _RANKS[to // 8]
    if p:
        uci += _PROMO[p]
    return uci


def select_move(model, state, num_simulations):
    search = GPUMCTS(model=model, device=next(model.parameters()).device)
    search.search(state, num_simulations=num_simulations)
    action = search.select_actions(temperature=0.0)
    return action, search.advance(action)


def play_match(white_model, black_model, num_simulations=10, max_moves=100):
    device = next(white_model.parameters()).device
    state = GPUChess(device, 1)
    move_history = []
    repetition = {}
    for _ in range(max_moves):
        key = int(state.state_hash()[0].item())
        repetition[key] = repetition.get(key, 0) + 1
        if repetition[key] >= 3:
            return {"result": 0, "termination": "threefold_repetition", "moves": move_history}
        terminal, _ = state.terminal_info()
        if bool(terminal[0].item()) or bool(state.insufficient_material()[0].item()):
            if bool(state.is_in_check()[0].item()) and not bool(state.legal_move_mask()[0].any().item()):
                result = -1 if not bool(state.turn[0].item()) else 1
                return {"result": result, "termination": "checkmate", "moves": move_history}
            return {"result": 0, "termination": "draw", "moves": move_history}
        model = black_model if bool(state.turn[0].item()) else white_model
        action, state = select_move(model, state, num_simulations)
        move_history.append(action_to_uci(int(action[0].item())))
    return {"result": None, "termination": "max_moves", "moves": move_history}


def evaluate_models(old_model, new_model, num_games=10, num_simulations=10, max_moves=100):
    new_wins = old_wins = draws = incomplete = 0
    terminations = {}
    for game in range(num_games):
        print(f"Evaluation game {game + 1}/{num_games}")
        if game % 2 == 0:
            result = play_match(new_model, old_model, num_simulations, max_moves)
            new_result = result["result"]
        else:
            result = play_match(old_model, new_model, num_simulations, max_moves)
            new_result = -result["result"] if result["result"] is not None else None
        if new_result == 1: new_wins += 1
        elif new_result == -1: old_wins += 1
        elif new_result == 0: draws += 1
        else: incomplete += 1
        terminations[result["termination"]] = terminations.get(result["termination"], 0) + 1
        print("Moves:", " ".join(result["moves"]))
        print("Result:", result["result"], "| Termination:", result["termination"])
    return {"new_wins": new_wins, "old_wins": old_wins, "draws": draws, "incomplete": incomplete, "terminations": terminations}

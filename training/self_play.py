"""GPU-first AlphaZero-style self-play.

The training self-play path uses the tensorized GPU chess engine and GPU MCTS
when the model is on CUDA.  A small legacy CPU fallback is retained for local
CPU use and for tools that still rely on python-chess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


@dataclass
class TrainingSample:
    state: np.ndarray
    policy: np.ndarray
    player: int


@dataclass
class SelfPlayResult:
    training_data: list
    result: int | None
    termination: str
    moves_played: int
    completed: bool


class SelfPlayGame:
    def __init__(self):
        self.samples = []

    def add_position(self, state: np.ndarray, policy: np.ndarray, player: int):
        self.samples.append(
            TrainingSample(
                state=np.asarray(state, dtype=np.float32),
                policy=np.asarray(policy, dtype=np.float32),
                player=int(player),
            )
        )

    def get_training_data(self, result):
        data = []
        for sample in self.samples:
            if result == 0:
                value = 0.0
            elif result == 1:
                value = float(sample.player)
            else:
                value = float(-sample.player)
            data.append((sample.state, sample.policy, value))
        return data


def _terminal_result(states: GPUChess, local_index: int) -> tuple[int, str, bool]:
    """Return result/termination for one terminal GPU state."""
    one = states.select(torch.tensor([local_index], device=states.device))
    legal = one.legal_move_mask()[0]
    in_check = bool(one.is_in_check()[0].item())
    if not bool(legal.any().item()):
        if in_check:
            # Side to move has been checkmated.
            result = -1 if bool(one.turn[0].item()) is False else 1
            return result, "CHECKMATE", True
        return 0, "STALEMATE", True
    if bool((one.halfmove_clock[0] >= 100).item()):
        return 0, "FIFTY_MOVE", True
    if bool(one.insufficient_material()[0].item()):
        return 0, "INSUFFICIENT_MATERIAL", True
    return 0, "UNKNOWN", False


def _play_games_gpu(
    model,
    num_games=8,
    num_simulations=100,
    max_moves=200,
    temperature=1.0,
    temperature_moves=20,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    batch_size=128,
):
    if num_games <= 0:
        return []
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("GPU self-play requires a CUDA model")

    # batch_size is retained for API compatibility. GPU MCTS already batches
    # one leaf per active game; the model sees the complete active batch.
    _ = batch_size
    states = GPUChess(device, num_games)
    games = [SelfPlayGame() for _ in range(num_games)]
    move_numbers = torch.ones((num_games,), dtype=torch.int32, device=device)
    active = torch.ones((num_games,), dtype=torch.bool, device=device)
    results: list[Optional[SelfPlayResult]] = [None] * num_games

    # Repetition tracking is intentionally outside the tensorized MCTS state.
    # It is small host-side metadata and does not participate in the hot chess
    # move-generation/search path.  The position key is derived from GPU state.
    repetition = [dict() for _ in range(num_games)]

    round_no = 0
    while bool(active.any().item()):
        round_no += 1
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        active_states = states.select(active_idx)

        # Record repetitions for the current actual game positions.
        keys = active_states.state_hash().detach().cpu().tolist()
        for local, global_idx in enumerate(active_idx.detach().cpu().tolist()):
            key = int(keys[local])
            repetition[global_idx][key] = repetition[global_idx].get(key, 0) + 1
            if repetition[global_idx][key] >= 3:
                results[global_idx] = SelfPlayResult([], 0, "THREEFOLD_REPETITION", len(games[global_idx].samples), True)
                active[global_idx] = False

        if not bool(active.any().item()):
            break

        # Refresh the active set after possible repetition termination.
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        active_states = states.select(active_idx)

        # Enforce maximum move count without creating a partial training game.
        too_long = move_numbers[active_idx] > max_moves
        if bool(too_long.any().item()):
            long_global = active_idx[too_long]
            for gi in long_global.detach().cpu().tolist():
                results[gi] = SelfPlayResult([], None, "MAX_MOVES", len(games[gi].samples), False)
            active[long_global] = False
            if not bool(active.any().item()):
                break
            active_idx = torch.nonzero(active, as_tuple=False).flatten()
            active_states = states.select(active_idx)

        search = GPUMCTS(
            model=model,
            device=device,
        )
        search.search(active_states, num_simulations=num_simulations)
        search.add_dirichlet_noise(dirichlet_alpha, dirichlet_epsilon)

        # Training targets are raw normalized visit counts; temperature only
        # affects the action sampled from that distribution.
        policies = search.root_visit_policy()
        current_temperature = temperature if round_no <= temperature_moves else 0.0
        actions = search.select_actions(current_temperature)
        next_states = search.advance(actions)

        # Store state/policy before the move. This matches the original
        # AlphaZero-style training target convention.
        state_batch = active_states.to_model_input().detach().cpu().numpy()
        policy_batch = policies.detach().cpu().numpy()
        players = (~active_states.turn).long().detach().cpu().tolist()
        global_indices = active_idx.detach().cpu().tolist()
        for local, gi in enumerate(global_indices):
            games[gi].add_position(state_batch[local], policy_batch[local], 1 if players[local] else -1)

        # Advance actual game states on GPU.
        states.pieces[active_idx] = next_states.pieces
        states.turn[active_idx] = next_states.turn
        states.castling[active_idx] = next_states.castling
        states.ep_square[active_idx] = next_states.ep_square
        states.halfmove_clock[active_idx] = next_states.halfmove_clock
        states.fullmove_number[active_idx] = next_states.fullmove_number
        move_numbers[active_idx] += 1

        # Determine which games have ended after the move.
        terminal, _ = next_states.terminal_info()
        terminal |= next_states.insufficient_material()
        term_local = torch.nonzero(terminal, as_tuple=False).flatten()
        for local in term_local.detach().cpu().tolist():
            gi = global_indices[local]
            result, termination, completed = _terminal_result(next_states, local)
            results[gi] = SelfPlayResult(
                training_data=games[gi].get_training_data(result) if completed else [],
                result=result if completed else None,
                termination=termination,
                moves_played=len(games[gi].samples),
                completed=completed,
            )
            active[gi] = False

        # Threefold is checked on the next loop using the new actual positions.

        if round_no % 10 == 0:
            print(f"GPU self-play round {round_no} | active games: {int(active.sum().item())}/{num_games}")

    print("\n" + "=" * 60)
    print("GPU SELF-PLAY COMPLETE")
    print("=" * 60)
    for i, result in enumerate(results):
        if result is None:
            result = SelfPlayResult([], None, "UNKNOWN", len(games[i].samples), False)
            results[i] = result
        print(
            f"Game {i + 1}: moves={result.moves_played} | "
            f"result={result.result} | termination={result.termination} | "
            f"completed={result.completed}"
        )
    print("Training samples:", sum(len(r.training_data) for r in results if r is not None))
    return results


def play_games(
    model,
    num_games=8,
    num_simulations=100,
    max_moves=200,
    temperature=1.0,
    temperature_moves=20,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    batch_size=128,
):
    """Run GPU-native self-play when the model is CUDA; otherwise use legacy path."""
    if next(model.parameters()).device.type == "cuda":
        return _play_games_gpu(
            model=model,
            num_games=num_games,
            num_simulations=num_simulations,
            max_moves=max_moves,
            temperature=temperature,
            temperature_moves=temperature_moves,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            batch_size=batch_size,
        )

    # CPU fallback imports are lazy so CUDA runs never import python-chess.
    from training.self_play_legacy import play_games as legacy_play_games
    return legacy_play_games(
        model=model,
        num_games=num_games,
        num_simulations=num_simulations,
        max_moves=max_moves,
        temperature=temperature,
        temperature_moves=temperature_moves,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        batch_size=batch_size,
    )


def play_game(
    model,
    num_simulations=100,
    max_moves=200,
    temperature=1.0,
    temperature_moves=20,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    batch_size=128,
):
    return play_games(
        model=model,
        num_games=1,
        num_simulations=num_simulations,
        max_moves=max_moves,
        temperature=temperature,
        temperature_moves=temperature_moves,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        batch_size=batch_size,
    )[0]

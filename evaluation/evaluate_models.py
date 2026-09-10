import sys
import os

import torch
import chess


# ==========================================================
# MAKE PROJECT ROOT IMPORTABLE
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ==========================================================
# IMPORTS
# ==========================================================

from model.chess_net import ChessNet
from environment.action_encoder import ActionEncoder
from mcts.node import Node
from mcts.mcts import MCTS


# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_GAMES = 10

NUM_SIMULATIONS = 50

MAX_MOVES = 300

EVALUATION_TEMPERATURE = 0.0


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# CHECKPOINTS
# ==========================================================

PRETRAINED_CHECKPOINT = (
    "/content/pretrained_phase1.pt"
)

RL_CHECKPOINT = (
    "/content/rl_iteration_1.pt"
)


# ==========================================================
# ACTION ENCODER
# ==========================================================

action_encoder = ActionEncoder()

print("Evaluation device:", DEVICE)
print("Action space size:", action_encoder.size())


# ==========================================================
# CREATE MODEL
# ==========================================================

def create_model():

    model = ChessNet(
        action_space_size=action_encoder.size()
    )

    model.to(DEVICE)
    model.eval()

    return model


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(
    model,
    checkpoint_path
):

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return checkpoint


# ==========================================================
# CREATE MODELS
# ==========================================================

pretrained_model = create_model()
rl_model = create_model()


# ==========================================================
# LOAD PHASE 1 MODEL
# ==========================================================

pretrained_checkpoint = load_model(
    pretrained_model,
    PRETRAINED_CHECKPOINT
)

print(
    "\nLoaded Phase 1 pretrained model."
)

print(
    "Checkpoint:",
    PRETRAINED_CHECKPOINT
)


# ==========================================================
# LOAD RL ITERATION 1 MODEL
# ==========================================================

rl_checkpoint = load_model(
    rl_model,
    RL_CHECKPOINT
)

print(
    "\nLoaded RL Iteration 1 model."
)

print(
    "Checkpoint:",
    RL_CHECKPOINT
)


# ==========================================================
# CHECKPOINT INFORMATION
# ==========================================================

print("\nRL checkpoint information:")

for key in [
    "iteration",
    "previous_checkpoint",
    "completed_games",
    "incomplete_games",
    "white_wins",
    "black_wins",
    "draws",
    "new_samples",
    "num_simulations",
    "training_steps"
]:

    if key in rl_checkpoint:

        print(
            f"{key}:",
            rl_checkpoint[key]
        )


# ==========================================================
# GET MOVE USING MCTS
# ==========================================================

def get_move(
    model,
    board
):

    mcts = MCTS(
        model=model,
        action_encoder=action_encoder
    )

    root = Node(board)

    mcts.search(
        root,
        num_simulations=NUM_SIMULATIONS
    )

    if not root.children:

        raise RuntimeError(
            "MCTS root has no children."
        )

    # ------------------------------------------------------
    # Deterministic evaluation
    # ------------------------------------------------------

    if EVALUATION_TEMPERATURE <= 0:

        move, child = mcts.select_action(root)

        if move is None:

            raise RuntimeError(
                "MCTS failed to select a move."
            )

        return move

    # ------------------------------------------------------
    # Stochastic evaluation
    # ------------------------------------------------------

    return mcts.select_action_with_temperature(
        root,
        temperature=EVALUATION_TEMPERATURE
    )


# ==========================================================
# PLAY ONE GAME
# ==========================================================

def play_game(
    white_model,
    black_model
):

    board = chess.Board()

    moves = 0

    while not board.is_game_over(
        claim_draw=True
    ):

        # --------------------------------------------------
        # Maximum move safety limit
        # --------------------------------------------------

        if moves >= MAX_MOVES:

            return {
                "result": None,
                "termination": "MAX_MOVES",
                "moves": moves
            }

        # --------------------------------------------------
        # Select model
        # --------------------------------------------------

        if board.turn == chess.WHITE:

            model = white_model

        else:

            model = black_model

        # --------------------------------------------------
        # Get move
        # --------------------------------------------------

        move = get_move(
            model,
            board
        )

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if move not in board.legal_moves:

            raise RuntimeError(
                f"Illegal move returned by MCTS: {move}"
            )

        board.push(move)

        moves += 1

    # ======================================================
    # GAME TERMINATED
    # ======================================================

    outcome = board.outcome(
        claim_draw=True
    )

    if outcome is None:

        return {
            "result": "1/2-1/2",
            "termination": "UNKNOWN",
            "moves": moves
        }

    # ------------------------------------------------------
    # Determine result
    # ------------------------------------------------------

    if outcome.winner == chess.WHITE:

        result = "1-0"

    elif outcome.winner == chess.BLACK:

        result = "0-1"

    else:

        result = "1/2-1/2"

    return {
        "result": result,
        "termination": str(
            outcome.termination
        ),
        "moves": moves
    }


# ==========================================================
# EVALUATE TWO MODELS
# ==========================================================

def evaluate_models(
    model_a,
    model_b,
    name_a,
    name_b
):

    a_wins = 0
    b_wins = 0
    true_draws = 0
    truncated = 0

    termination_counts = {}

    total_moves = 0

    # ======================================================
    # HEADER
    # ======================================================

    print(
        "\n=============================="
    )

    print(
        f"{name_a.upper()} VS {name_b.upper()}"
    )

    print(
        "=============================="
    )

    print(
        "Games:",
        NUM_GAMES
    )

    print(
        "MCTS simulations:",
        NUM_SIMULATIONS
    )

    print(
        "Evaluation temperature:",
        EVALUATION_TEMPERATURE
    )

    # ======================================================
    # PLAY GAMES
    # ======================================================

    for game_number in range(
        1,
        NUM_GAMES + 1
    ):

        # --------------------------------------------------
        # Alternate colors
        # --------------------------------------------------

        if game_number % 2 == 1:

            white_model = model_a
            black_model = model_b

            a_color = "White"

        else:

            white_model = model_b
            black_model = model_a

            a_color = "Black"

        print(
            f"\nGame {game_number}/{NUM_GAMES}"
        )

        print(
            f"{name_a}: {a_color}"
        )

        # --------------------------------------------------
        # Play
        # --------------------------------------------------

        result = play_game(
            white_model,
            black_model
        )

        print(
            "Result:",
            result["result"]
        )

        print(
            "Termination:",
            result["termination"]
        )

        print(
            "Moves:",
            result["moves"]
        )

        total_moves += result["moves"]

        # ==================================================
        # SCORE
        # ==================================================

        if result["result"] is None:

            truncated += 1

        elif result["result"] == "1/2-1/2":

            true_draws += 1

        elif (
            result["result"] == "1-0"
            and a_color == "White"
        ):

            a_wins += 1

        elif (
            result["result"] == "0-1"
            and a_color == "Black"
        ):

            a_wins += 1

        else:

            b_wins += 1

        # ==================================================
        # TERMINATION
        # ==================================================

        termination = result["termination"]

        termination_counts[termination] = (
            termination_counts.get(
                termination,
                0
            ) + 1
        )

    # ======================================================
    # RESULTS
    # ======================================================

    print(
        "\n=============================="
    )

    print(
        "EVALUATION RESULTS"
    )

    print(
        "=============================="
    )

    print(
        f"{name_a} wins:",
        a_wins
    )

    print(
        f"{name_b} wins:",
        b_wins
    )

    print(
        "True draws:",
        true_draws
    )

    print(
        "Truncated:",
        truncated
    )

    print(
        "Total moves:",
        total_moves
    )

    # ======================================================
    # COMPLETED GAMES
    # ======================================================

    completed_games = (
        a_wins
        + b_wins
        + true_draws
    )

    print(
        "\nCompleted games:",
        completed_games
    )

    # ======================================================
    # SCORE
    # ======================================================

    if completed_games > 0:

        a_score = (
            a_wins
            + 0.5 * true_draws
        ) / completed_games

        b_score = (
            b_wins
            + 0.5 * true_draws
        ) / completed_games

    else:

        a_score = 0.0
        b_score = 0.0

    print(
        "Score among completed games:"
    )

    print(
        f"{name_a}: {a_score:.3f}"
    )

    print(
        f"{name_b}: {b_score:.3f}"
    )

    # ======================================================
    # TRUNCATION RATE
    # ======================================================

    truncation_rate = (
        truncated / NUM_GAMES
        if NUM_GAMES > 0
        else 0.0
    )

    print(
        "\nTruncation rate:",
        f"{truncation_rate:.2%}"
    )

    # ======================================================
    # TERMINATION REASONS
    # ======================================================

    print(
        "\nTermination reasons:"
    )

    for reason, count in (
        termination_counts.items()
    ):

        print(
            reason,
            ":",
            count
        )

    # ======================================================
    # AVERAGE GAME LENGTH
    # ======================================================

    if NUM_GAMES > 0:

        average_moves = (
            total_moves / NUM_GAMES
        )

        print(
            "\nAverage game length:",
            f"{average_moves:.1f} moves"
        )

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": true_draws,
        "truncated": truncated,
        "completed_games": completed_games,
        "a_score": a_score,
        "b_score": b_score,
        "termination_counts":
            termination_counts
    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    evaluate_models(
        pretrained_model,
        rl_model,
        "Phase 1 Pretrained",
        "RL Iteration 1"
    )
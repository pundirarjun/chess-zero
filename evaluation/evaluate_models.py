import sys
import os
import random

import numpy as np
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

    sys.path.append(
        PROJECT_ROOT
    )


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

# ----------------------------------------------------------
# Number of evaluation games per matchup
# ----------------------------------------------------------

NUM_GAMES = 10


# ----------------------------------------------------------
# MCTS simulations per move
# ----------------------------------------------------------

NUM_SIMULATIONS = 100


# ----------------------------------------------------------
# MCTS batch size
#
# Evaluation uses normal MCTS here, so this is not currently
# used by get_move(). It is kept available if we later switch
# evaluation to batched MCTS.
# ----------------------------------------------------------

MCTS_BATCH_SIZE = 64


# ----------------------------------------------------------
# Maximum number of plies per game
# ----------------------------------------------------------

MAX_MOVES = 300


# ----------------------------------------------------------
# Evaluation temperature
#
# 0.0 = deterministic highest-visit-count move.
#
# IMPORTANT:
# Evaluation should normally be deterministic.
# ----------------------------------------------------------

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
# ACTION ENCODER
# ==========================================================

action_encoder = ActionEncoder()


print(
    "Evaluation device:",
    DEVICE
)

print(
    "Action space size:",
    action_encoder.size()
)


# ==========================================================
# CREATE MODEL
# ==========================================================

def create_model():

    model = ChessNet(
        action_space_size=action_encoder.size()
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ==========================================================
# LOAD MODEL CHECKPOINT
# ==========================================================

def load_model(
    model,
    checkpoint_path
):

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE,

        weights_only=False

    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        DEVICE
    )

    model.eval()

    return checkpoint


# ==========================================================
# CREATE MODELS
# ==========================================================

random_model = create_model()

pretrained_model = create_model()

rl_model = create_model()


# ==========================================================
# LOAD PRETRAINED MODEL
# ==========================================================

pretrained_checkpoint = load_model(

    pretrained_model,

    "checkpoints/pretrained_1000_games.pt"

)

print(
    "\nLoaded pretrained model."
)

if "iteration" in pretrained_checkpoint:

    print(
        "Pretrained checkpoint iteration:",
        pretrained_checkpoint["iteration"]
    )


# ==========================================================
# LOAD RL MODEL
# ==========================================================

rl_checkpoint = load_model(

    rl_model,

    "checkpoints/rl_iteration_4.pt"

)

print(
    "Loaded RL model."
)

if "iteration" in rl_checkpoint:

    print(
        "RL checkpoint iteration:",
        rl_checkpoint["iteration"]
    )


# ==========================================================
# ENSURE EVAL MODE
# ==========================================================

random_model.eval()

pretrained_model.eval()

rl_model.eval()


# ==========================================================
# GET MOVE USING MCTS
# ==========================================================

def get_move(
    model,
    board
):

    # ------------------------------------------------------
    # Create MCTS object
    # ------------------------------------------------------

    mcts = MCTS(

        model=model,

        action_encoder=action_encoder

    )

    # ------------------------------------------------------
    # Create root
    # ------------------------------------------------------

    root = Node(
        board
    )

    # ------------------------------------------------------
    # Run exactly NUM_SIMULATIONS simulations
    #
    # IMPORTANT:
    #
    # Do NOT manually subtract 1.
    #
    # MCTS.search() already handles root initialization.
    # ------------------------------------------------------

    mcts.search(

        root,

        num_simulations=NUM_SIMULATIONS

    )

    # ------------------------------------------------------
    # Safety check
    # ------------------------------------------------------

    if not root.children:

        raise RuntimeError(
            "MCTS root has no children."
        )

    # ------------------------------------------------------
    # Deterministic evaluation
    # ------------------------------------------------------

    if EVALUATION_TEMPERATURE <= 0:

        move, child = (
            mcts.select_action(
                root
            )
        )

        if move is None:

            raise RuntimeError(
                "MCTS failed to select a move."
            )

        return move

    # ------------------------------------------------------
    # Optional stochastic evaluation
    #
    # Normally not used.
    # ------------------------------------------------------

    return (
        mcts.select_action_with_temperature(

            root,

            temperature=EVALUATION_TEMPERATURE

        )
    )


# ==========================================================
# MOVE RANDOMNESS TEST
# ==========================================================

def test_move_determinism(
    model,
    model_name
):

    print(
        "\n=============================="
    )

    print(
        "MOVE DETERMINISM TEST"
    )

    print(
        "=============================="
    )

    print(
        "Model:",
        model_name
    )

    board = chess.Board()

    move_counts = {}

    for i in range(10):

        move = get_move(

            model,

            board

        )

        move_name = move.uci()

        move_counts[
            move_name
        ] = (
            move_counts.get(
                move_name,
                0
            ) + 1
        )

        print(
            f"Test {i + 1}: {move_name}"
        )

    print(
        "\nMove frequencies:"
    )

    for move, count in sorted(

        move_counts.items(),

        key=lambda x: x[1],

        reverse=True

    ):

        print(
            f"{move}: {count}"
        )

    # ------------------------------------------------------
    # Determinism check
    # ------------------------------------------------------

    if len(move_counts) == 1:

        print(
            "\nDeterminism test: PASS"
        )

    else:

        print(
            "\nDeterminism test: WARNING"
        )

        print(
            "Different moves were selected."
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
        # Safety limit
        # --------------------------------------------------

        if moves >= MAX_MOVES:

            return {

                "result": None,

                "termination":
                    "MAX_MOVES",

                "moves":
                    moves

            }

        # --------------------------------------------------
        # Select model
        # --------------------------------------------------

        if board.turn == chess.WHITE:

            model = white_model

        else:

            model = black_model

        # --------------------------------------------------
        # Get MCTS move
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

                f"Illegal move returned by MCTS: "
                f"{move}"

            )

        # --------------------------------------------------
        # Play move
        # --------------------------------------------------

        board.push(
            move
        )

        moves += 1

    # ======================================================
    # GAME TERMINATED
    # ======================================================

    outcome = board.outcome(
        claim_draw=True
    )

    if outcome is None:

        return {

            "result":
                "1/2-1/2",

            "termination":
                "UNKNOWN",

            "moves":
                moves

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

        "result":
            result,

        "termination":
            str(
                outcome.termination
            ),

        "moves":
            moves

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
        f"{name_a.upper()} VS "
        f"{name_b.upper()}"
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
    # GAMES
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
        # Play game
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
        # SCORE RESULT
        # ==================================================

        if result["result"] is None:

            # MAX_MOVES

            truncated += 1

        elif result["result"] == "1/2-1/2":

            # Actual chess draw

            true_draws += 1

        elif (

            result["result"] == "1-0"

            and

            a_color == "White"

        ):

            a_wins += 1

        elif (

            result["result"] == "0-1"

            and

            a_color == "Black"

        ):

            a_wins += 1

        else:

            b_wins += 1

        # ==================================================
        # TERMINATION COUNT
        # ==================================================

        termination = result[
            "termination"
        ]

        termination_counts[
            termination
        ] = (
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
        "\nCompleted games:",
        completed_games
    )

    print(
        "Score among completed games:"
    )

    print(
        f"{name_a}: "
        f"{a_score:.3f}"
    )

    print(
        f"{name_b}: "
        f"{b_score:.3f}"
    )

    # ======================================================
    # TRUNCATION RATE
    # ======================================================

    if NUM_GAMES > 0:

        truncation_rate = (
            truncated
            / NUM_GAMES
        )

    else:

        truncation_rate = 0.0

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
            total_moves
            / NUM_GAMES
        )

        print(
            "\nAverage game length:",
            f"{average_moves:.1f} moves"
        )

    return {

        "a_wins":
            a_wins,

        "b_wins":
            b_wins,

        "draws":
            true_draws,

        "truncated":
            truncated,

        "completed_games":
            completed_games,

        "a_score":
            a_score,

        "b_score":
            b_score,

        "termination_counts":
            termination_counts

    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # 1. TEST RL MODEL DETERMINISM
    # ======================================================

    test_move_determinism(

        rl_model,

        "RL Iteration 4"

    )

    # ======================================================
    # 2. PRETRAINED VS RANDOM
    # ======================================================

    evaluate_models(

        pretrained_model,

        random_model,

        "Pretrained",

        "Random"

    )

    # ======================================================
    # 3. RL VS RANDOM
    # ======================================================

    evaluate_models(

        rl_model,

        random_model,

        "RL Iteration 4",

        "Random"

    )

    # ======================================================
    # 4. RL VS PRETRAINED
    # ======================================================

    evaluate_models(

        rl_model,

        pretrained_model,

        "RL Iteration 4",

        "Pretrained"

    )
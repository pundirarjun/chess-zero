import sys
import os

# ==================================================
# Make project root importable
# ==================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import random
import numpy as np
import torch
import chess

from model.chess_net import ChessNet
from environment.action_encoder import ActionEncoder
from mcts.node import Node
from mcts.mcts import MCTS


# ==================================================
# CONFIGURATION
# ==================================================

NUM_GAMES = 10

NUM_SIMULATIONS = 25

MAX_MOVES = 300

# Temperature used ONLY during evaluation.
#
# Lower = more deterministic / stronger preference
# for the highest visit-count move.
#
# Higher = more exploration.
EVALUATION_TEMPERATURE = 0.25


# ==================================================
# SETUP
# ==================================================

action_encoder = ActionEncoder()


# ==================================================
# RANDOM MODEL
# ==================================================

random_model = ChessNet(
    action_space_size=action_encoder.size()
)


# ==================================================
# PRETRAINED MODEL
# ==================================================

pretrained_model = ChessNet(
    action_space_size=action_encoder.size()
)


# ==================================================
# RL MODEL
# ==================================================

rl_model = ChessNet(
    action_space_size=action_encoder.size()
)


# ==================================================
# LOAD PRETRAINED CHECKPOINT
# ==================================================

pretrained_checkpoint = torch.load(
    "checkpoints/pretrained_1000_games.pt",
    map_location="cpu"
)

pretrained_model.load_state_dict(
    pretrained_checkpoint[
        "model_state_dict"
    ]
)


# ==================================================
# LOAD RL ITERATION 4 CHECKPOINT
# ==================================================

rl_checkpoint = torch.load(
    "checkpoints/rl_iteration_4.pt",
    map_location="cpu"
)

rl_model.load_state_dict(
    rl_checkpoint[
        "model_state_dict"
    ]
)


# ==================================================
# EVAL MODE
# ==================================================

random_model.eval()

pretrained_model.eval()

rl_model.eval()


# ==================================================
# MCTS MOVE
# ==================================================

def get_move(model, board):

    mcts = MCTS(
        model=model,
        action_encoder=action_encoder
    )

    root = Node(board)

    # --------------------------------------------------
    # Expand root
    # --------------------------------------------------

    root.expand(
        model,
        action_encoder
    )

    # --------------------------------------------------
    # Run MCTS simulations
    # --------------------------------------------------

    for _ in range(
        NUM_SIMULATIONS - 1
    ):

        mcts.run_simulation(
            root
        )

    # --------------------------------------------------
    # Get visit counts
    # --------------------------------------------------

    children = root.children

    if not children:

        raise RuntimeError(
            "MCTS root has no children."
        )

    moves = []

    visit_counts = []

    for move, child in children.items():

        moves.append(move)

        visit_counts.append(
            max(
                0,
                child.visit_count
            )
        )

    visit_counts = np.asarray(
        visit_counts,
        dtype=np.float64
    )

    # --------------------------------------------------
    # Safety check
    # --------------------------------------------------

    total_visits = visit_counts.sum()

    if total_visits <= 0:

        # This should almost never happen.
        # Fall back to a random legal move.

        legal_moves = list(
            board.legal_moves
        )

        return random.choice(
            legal_moves
        )

    # ==================================================
    # TEMPERATURE SAMPLING
    # ==================================================

    temperature = (
        EVALUATION_TEMPERATURE
    )

    if temperature <= 0:

        # Pure greedy selection

        selected_index = int(
            np.argmax(
                visit_counts
            )
        )

    else:

        # AlphaZero-style temperature transformation:
        #
        # probability ∝ visit_count^(1 / temperature)

        probabilities = (
            visit_counts
            ** (
                1.0 / temperature
            )
        )

        probability_sum = (
            probabilities.sum()
        )

        if probability_sum <= 0:

            probabilities = (
                np.ones_like(
                    visit_counts
                )
                / len(visit_counts)
            )

        else:

            probabilities = (
                probabilities
                / probability_sum
            )

        # --------------------------------------------------
        # Explicit stochastic sampling
        # --------------------------------------------------

        selected_index = np.random.choice(
            len(moves),
            p=probabilities
        )

    return moves[
        selected_index
    ]


# ==================================================
# MOVE RANDOMNESS TEST
# ==================================================

def test_move_randomness(model):

    print(
        "\n=============================="
    )

    print(
        "MOVE RANDOMNESS TEST"
    )

    print(
        "=============================="
    )

    board = chess.Board()

    move_counts = {}

    for i in range(10):

        move = get_move(
            model,
            board
        )

        move_name = (
            move.uci()
        )

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


# ==================================================
# PLAY GAME
# ==================================================

def play_game(
    white_model,
    black_model
):

    board = chess.Board()

    moves = 0

    while not board.is_game_over(
        claim_draw=True
    ):

        # ------------------------------------------
        # Safety limit
        # ------------------------------------------

        if moves >= MAX_MOVES:

            return {
                "result": None,
                "termination": "MAX_MOVES",
                "moves": moves
            }

        # ------------------------------------------
        # Select model
        # ------------------------------------------

        if board.turn == chess.WHITE:

            model = white_model

        else:

            model = black_model

        # ------------------------------------------
        # Get MCTS move
        # ------------------------------------------

        move = get_move(
            model,
            board
        )

        # ------------------------------------------
        # Safety check
        # ------------------------------------------

        if move not in board.legal_moves:

            raise RuntimeError(
                f"Illegal move returned by MCTS: {move}"
            )

        # ------------------------------------------
        # Play move
        # ------------------------------------------

        board.push(move)

        moves += 1

    # ==================================================
    # GAME TERMINATED
    # ==================================================

    outcome = board.outcome(
        claim_draw=True
    )

    if outcome is None:

        result = "1/2-1/2"

        termination = "UNKNOWN"

    else:

        if outcome.winner == chess.WHITE:

            result = "1-0"

        elif outcome.winner == chess.BLACK:

            result = "0-1"

        else:

            result = "1/2-1/2"

        termination = str(
            outcome.termination
        )

    return {
        "result": result,
        "termination": termination,
        "moves": moves
    }


# ==================================================
# EVALUATE TWO MODELS
# ==================================================

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

    print(
        "\n=============================="
    )

    print(
        f"{name_a.upper()} VS {name_b.upper()}"
    )

    print(
        "=============================="
    )

    # ==================================================
    # GAMES
    # ==================================================

    for game_number in range(
        1,
        NUM_GAMES + 1
    ):

        # ------------------------------------------
        # Alternate colors
        # ------------------------------------------

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
            f"{name_a}:",
            a_color
        )

        # ------------------------------------------
        # Play
        # ------------------------------------------

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

    # ==================================================
    # RESULTS
    # ==================================================

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

    # ==================================================
    # SCORE
    # ==================================================

    completed_games = (
        a_wins
        + b_wins
        + true_draws
    )

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
        "\nScore among completed games:"
    )

    print(
        f"{name_a}: {a_score:.3f}"
    )

    print(
        f"{name_b}: {b_score:.3f}"
    )

    # ==================================================
    # TERMINATIONS
    # ==================================================

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


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    # ==================================================
    # 1. TEST RANDOMNESS
    # ==================================================

    test_move_randomness(
        rl_model
    )

    # ==================================================
    # 2. PRETRAINED VS RANDOM
    # ==================================================

    evaluate_models(
        pretrained_model,
        random_model,
        "Pretrained",
        "Random"
    )

    # ==================================================
    # 3. RL ITERATION 4 VS RANDOM
    # ==================================================

    evaluate_models(
        rl_model,
        random_model,
        "RL Iteration 4",
        "Random"
    )

    # ==================================================
    # 4. RL ITERATION 4 VS PRETRAINED
    # ==================================================

    evaluate_models(
        rl_model,
        pretrained_model,
        "RL Iteration 4",
        "Pretrained"
    )
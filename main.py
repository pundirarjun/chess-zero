import sys
import os

# ==========================================================
# MAKE PROJECT ROOT IMPORTABLE
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:

    sys.path.append(
        PROJECT_ROOT
    )


# ==========================================================
# IMPORTS
# ==========================================================

import torch
import chess

from model.chess_net import ChessNet

from environment.action_encoder import ActionEncoder

from mcts.node import Node
from mcts.mcts import MCTS


# ==========================================================
# CONFIGURATION
# ==========================================================

CHECKPOINT = (
    "checkpoints/rl_iteration_4.pt"
)

SIMULATION_COUNTS = [
    25,
    50,
    100,
    250
]


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


# ==========================================================
# ACTION ENCODER
# ==========================================================

action_encoder = ActionEncoder()

print(
    "Action space:",
    action_encoder.size()
)


# ==========================================================
# CREATE MODEL
# ==========================================================

model = ChessNet(

    action_space_size=
        action_encoder.size()

)

model.to(
    DEVICE
)


# ==========================================================
# LOAD CHECKPOINT
# ==========================================================

checkpoint = torch.load(

    CHECKPOINT,

    map_location=DEVICE,

    weights_only=False

)

model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]

)

model.eval()


print(
    "Checkpoint loaded:",
    CHECKPOINT
)

if "iteration" in checkpoint:

    print(
        "Checkpoint iteration:",
        checkpoint["iteration"]
    )


# ==========================================================
# TACTICAL POSITION
# ==========================================================

board = chess.Board()

board.clear()


# ==========================================================
# BLACK KING
# ==========================================================

board.set_piece_at(

    chess.G8,

    chess.Piece(
        chess.KING,
        chess.BLACK
    )

)


# ==========================================================
# BLACK QUEEN
# ==========================================================

board.set_piece_at(

    chess.D5,

    chess.Piece(
        chess.QUEEN,
        chess.BLACK
    )

)


# ==========================================================
# BLACK PAWNS
# ==========================================================

board.set_piece_at(

    chess.F7,

    chess.Piece(
        chess.PAWN,
        chess.BLACK
    )

)

board.set_piece_at(

    chess.G7,

    chess.Piece(
        chess.PAWN,
        chess.BLACK
    )

)

board.set_piece_at(

    chess.H7,

    chess.Piece(
        chess.PAWN,
        chess.BLACK
    )

)


# ==========================================================
# WHITE QUEEN
# ==========================================================

board.set_piece_at(

    chess.E4,

    chess.Piece(
        chess.QUEEN,
        chess.WHITE
    )

)


# ==========================================================
# WHITE KING
# ==========================================================

board.set_piece_at(

    chess.G1,

    chess.Piece(
        chess.KING,
        chess.WHITE
    )

)


# ==========================================================
# WHITE PAWNS
# ==========================================================

board.set_piece_at(

    chess.F2,

    chess.Piece(
        chess.PAWN,
        chess.WHITE
    )

)

board.set_piece_at(

    chess.G2,

    chess.Piece(
        chess.PAWN,
        chess.WHITE
    )

)

board.set_piece_at(

    chess.H2,

    chess.Piece(
        chess.PAWN,
        chess.WHITE
    )

)


# ==========================================================
# GAME STATE
# ==========================================================

board.turn = chess.WHITE


# ==========================================================
# PRINT POSITION
# ==========================================================

print(
    "\n=============================="
)

print(
    "TACTICAL POSITION"
)

print(
    "=============================="
)

print(
    board
)


# ==========================================================
# BASIC POSITION CHECK
# ==========================================================

print(
    "\nLegal moves:",
    board.legal_moves.count()
)


# ==========================================================
# RUN MCTS TESTS
# ==========================================================

for simulations in SIMULATION_COUNTS:

    print(
        "\n=============================="
    )

    print(
        f"MCTS: {simulations} SIMULATIONS"
    )

    print(
        "=============================="
    )


    # ------------------------------------------------------
    # Create MCTS
    # ------------------------------------------------------

    mcts = MCTS(

        model=model,

        action_encoder=
            action_encoder

    )


    # ------------------------------------------------------
    # Create fresh root
    # ------------------------------------------------------

    root = Node(
        board
    )


    # ------------------------------------------------------
    # Run exactly N simulations
    # ------------------------------------------------------

    mcts.search(

        root,

        num_simulations=
            simulations

    )


    # ------------------------------------------------------
    # Check root
    # ------------------------------------------------------

    print(
        "Root visits:",
        root.visit_count
    )

    print(
        "Root children:",
        len(root.children)
    )


    # ------------------------------------------------------
    # Sort children
    # ------------------------------------------------------

    sorted_children = sorted(

        root.children.items(),

        key=lambda item:
            item[1].visit_count,

        reverse=True

    )


    # ======================================================
    # PRINT TOP MOVES
    # ======================================================

    print(
        "\nTop moves:"
    )

    for move, child in (
        sorted_children[:10]
    ):

        print(

            f"{move} | "

            f"visits: "
            f"{child.visit_count} | "

            f"value: "
            f"{child.value:.4f} | "

            f"prior: "
            f"{child.prior:.4f}"

        )


    # ======================================================
    # SELECT BEST MOVE
    # ======================================================

    best_move, best_child = (

        mcts.select_action(
            root
        )

    )


    print(
        "\nSelected move:",
        best_move
    )

    print(
        "Visits:",
        best_child.visit_count
    )

    print(
        "Value:",
        round(
            best_child.value,
            4
        )
    )

    print(
        "Prior:",
        round(
            best_child.prior,
            4
        )
    )
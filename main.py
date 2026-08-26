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

import torch
import chess

from model.chess_net import ChessNet
from environment.action_encoder import ActionEncoder
from mcts.node import Node
from mcts.mcts import MCTS


# ==================================================
# CONFIGURATION
# ==================================================

CHECKPOINT = "checkpoints/rl_iteration_4.pt"

SIMULATION_COUNTS = [
    25,
    50,
    100,
    250
]


# ==================================================
# LOAD MODEL
# ==================================================

action_encoder = ActionEncoder()

model = ChessNet(
    action_space_size=action_encoder.size()
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ==================================================
# TACTICAL POSITION
# ==================================================

board = chess.Board()

board.clear()

# Black king
board.set_piece_at(
    chess.G8,
    chess.Piece(
        chess.KING,
        chess.BLACK
    )
)

# Black queen
board.set_piece_at(
    chess.D5,
    chess.Piece(
        chess.QUEEN,
        chess.BLACK
    )
)

# Black pawns
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

# White queen
board.set_piece_at(
    chess.E4,
    chess.Piece(
        chess.QUEEN,
        chess.WHITE
    )
)

# White king
board.set_piece_at(
    chess.G1,
    chess.Piece(
        chess.KING,
        chess.WHITE
    )
)

# White pawns
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

# White to move
board.turn = chess.WHITE


# ==================================================
# PRINT POSITION
# ==================================================

print(
    "\n=============================="
)

print(
    "TACTICAL POSITION"
)

print(
    "=============================="
)

print(board)


# ==================================================
# RUN MCTS
# ==================================================

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

    mcts = MCTS(
        model=model,
        action_encoder=action_encoder
    )

    root = Node(board)

    # Expand root
    root.expand(
        model,
        action_encoder
    )

    # Run remaining simulations
    for _ in range(
        max(
            0,
            simulations - 1
        )
    ):

        mcts.run_simulation(
            root
        )

    # Sort by visits
    sorted_children = sorted(
        root.children.items(),
        key=lambda item: (
            item[1].visit_count
        ),
        reverse=True
    )

    print(
        "\nTop moves:"
    )

    for move, child in sorted_children[:10]:

        print(
            f"{move} | "
            f"visits: {child.visit_count} | "
            f"value: {child.value:.4f} | "
            f"prior: {child.prior:.4f}"
        )

    # Best move
    best_move, best_child = (
        mcts.select_action(root)
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
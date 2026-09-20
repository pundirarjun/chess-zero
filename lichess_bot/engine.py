import os
import sys
import chess
import torch

# ============================================================
# PROJECT PATH
# ============================================================

# lichess_bot/
#     engine.py
#
# Parent directory is the ChessAI project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT EXISTING CHESS AI COMPONENTS
# ============================================================

from model.chess_net import ChessNet
from mcts.mcts import MCTS
from environment.action_encoder import ActionEncoder


# ============================================================
# CONFIGURATION
# ============================================================

# Change this only if your checkpoint has a different name/location.
MODEL_PATH = '/kaggle/input/datasets/arjunthakur9999/checkpoints/rl_iteration_13.pt'

ACTION_SPACE_SIZE = 4544

# Number of MCTS simulations for every move.
# Start with 100 because this matches your evaluation setup.
NUM_SIMULATIONS = 100


# ============================================================
# ENGINE
# ============================================================

class ChessEngine:

    def __init__(
        self,
        model_path=MODEL_PATH,
        num_simulations=NUM_SIMULATIONS
    ):
        self.model_path = model_path
        self.num_simulations = num_simulations

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Engine device: {self.device}")

        # ----------------------------------------------------
        # Action Encoder
        # ----------------------------------------------------

        self.action_encoder = ActionEncoder()

        print(
            f"Action space size: "
            f"{self.action_encoder.size()}"
        )

        if self.action_encoder.size() != ACTION_SPACE_SIZE:
            raise ValueError(
                f"Action space mismatch! "
                f"Expected {ACTION_SPACE_SIZE}, "
                f"got {self.action_encoder.size()}"
            )

        # ----------------------------------------------------
        # Create Model
        # ----------------------------------------------------

        self.model = ChessNet(
            action_space_size=ACTION_SPACE_SIZE
        ).to(self.device)

        # ----------------------------------------------------
        # Load RL23 checkpoint
        # ----------------------------------------------------

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Checkpoint not found:\n{self.model_path}"
            )

        print(f"Loading checkpoint:\n{self.model_path}")

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )

        # Your checkpoints use model_state_dict.
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # ----------------------------------------------------
        # CUDA optimization
        # ----------------------------------------------------

        if self.device.type == "cuda":
            self.model.to(
                memory_format=torch.channels_last
            )

        # ----------------------------------------------------
        # Create MCTS
        # ----------------------------------------------------

        self.mcts = MCTS(
            model=self.model,
            action_encoder=self.action_encoder
        )

        print("RL23 engine loaded successfully.")

        # Print checkpoint information if available
        if isinstance(checkpoint, dict):

            if "iteration" in checkpoint:
                print(
                    f"Checkpoint iteration: "
                    f"{checkpoint['iteration']}"
                )

            if "completed_games" in checkpoint:
                print(
                    f"Completed games: "
                    f"{checkpoint['completed_games']}"
                )

    # ========================================================
    # CHOOSE MOVE
    # ========================================================

    def choose_move(self, board):
        """
        Choose a legal chess move for the given board.

        Parameters
        ----------
        board : chess.Board
            Current chess position.

        Returns
        -------
        chess.Move
            Selected legal move.
        """

        if board.is_game_over():
            return None

        print("\nPosition:")
        print(board)

        print(
            f"\nSearching with "
            f"{self.num_simulations} MCTS simulations..."
        )

        # ----------------------------------------------------
        # Existing MCTS
        # ----------------------------------------------------

        move = self.mcts.select_move(
            board,
            num_simulations=self.num_simulations
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if move not in board.legal_moves:
            raise RuntimeError(
                f"MCTS returned an illegal move: {move}"
            )

        print(f"Selected move: {move}")
        print(f"UCI: {move.uci()}")

        return move

    # ========================================================
    # FEN INTERFACE
    # ========================================================

    def choose_move_from_fen(self, fen):
        """
        Choose a move directly from a FEN position.

        Returns UCI move string.
        """

        board = chess.Board(fen)

        move = self.choose_move(board)

        if move is None:
            return None

        return move.uci()


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RL23 CHESS ENGINE TEST")
    print("=" * 60)

    engine = ChessEngine()

    # Start from normal chess starting position
    board = chess.Board()

    move = engine.choose_move(board)

    print("\n" + "=" * 60)
    print("ENGINE TEST RESULT")
    print("=" * 60)

    print(f"Move: {move}")
    print(f"UCI:  {move.uci()}")

    print("\nEngine test completed successfully.")
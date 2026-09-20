import chess
import time

from engine import ChessEngine


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SIMULATIONS = 100


# ============================================================
# DISPLAY
# ============================================================

def print_board(board):
    print("\n" + "=" * 60)
    print(board)
    print("=" * 60)

    print(f"FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")


# ============================================================
# HUMAN MOVE
# ============================================================

def get_human_move(board):
    while True:
        move_input = input("\nYour move (UCI, e.g. e2e4): ").strip()

        if move_input.lower() in ["quit", "exit", "resign"]:
            return None

        try:
            move = chess.Move.from_uci(move_input)
        except ValueError:
            print("Invalid format. Use UCI notation, e.g. e2e4 or g1f3.")
            continue

        if move not in board.legal_moves:
            print("Illegal move.")
            print("Legal moves:")
            print(" ".join(move.uci() for move in board.legal_moves))
            continue

        return move


# ============================================================
# GAME
# ============================================================

def play_game(human_color):
    print("\n" + "=" * 60)
    print("RL30 HUMAN VS AI")
    print("=" * 60)

    print(
        f"You are playing "
        f"{'White' if human_color == chess.WHITE else 'Black'}."
    )

    print("Type 'resign' to resign.")
    print("Moves must be entered in UCI format, e.g. e2e4.")
    print("=" * 60)

    # --------------------------------------------------------
    # Create engine
    # --------------------------------------------------------

    engine = ChessEngine(
        num_simulations=NUM_SIMULATIONS
    )

    board = chess.Board()

    # --------------------------------------------------------
    # Game loop
    # --------------------------------------------------------

    while not board.is_game_over():

        print_board(board)

        # ====================================================
        # HUMAN TURN
        # ====================================================

        if board.turn == human_color:

            move = get_human_move(board)

            if move is None:
                print("\nYou resigned.")
                print("Game over.")
                return

            print(f"\nYour move: {move.uci()}")

            board.push(move)

        # ====================================================
        # AI TURN
        # ====================================================

        else:

            print("\nAI is thinking...")

            start_time = time.time()

            move = engine.choose_move(board)

            thinking_time = time.time() - start_time

            if move is None:
                break

            print(f"\nAI move: {move.uci()}")
            print(f"Thinking time: {thinking_time:.2f} seconds")

            board.push(move)

    # ========================================================
    # GAME RESULT
    # ========================================================

    print_board(board)

    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)

    print(f"Result: {board.result()}")

    if board.is_checkmate():

        winner = "White" if board.turn == chess.BLACK else "Black"

        print(f"Checkmate!")
        print(f"Winner: {winner}")

    elif board.is_stalemate():
        print("Stalemate.")

    elif board.is_insufficient_material():
        print("Draw by insufficient material.")

    elif board.is_fifty_moves():
        print("Draw by fifty-move rule.")

    elif board.is_repetition():
        print("Draw by repetition.")

    else:
        print("Game ended.")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RL30 CHESS - HUMAN VS AI")
    print("=" * 60)

    while True:

        color = input(
            "\nChoose your color (white/black): "
        ).strip().lower()

        if color in ["white", "w"]:
            human_color = chess.WHITE
            break

        elif color in ["black", "b"]:
            human_color = chess.BLACK
            break

        else:
            print("Please enter 'white' or 'black'.")

    play_game(human_color)


if __name__ == "__main__":
    main()


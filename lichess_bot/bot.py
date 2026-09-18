import os
import sys
import time
import chess


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(
        0,
        os.path.dirname(os.path.abspath(__file__))
    )


# ============================================================
# IMPORTS
# ============================================================

from engine import ChessEngine
from lichess_client import LichessClient


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SIMULATIONS = 100


# ============================================================
# RL23 LICHESS BOT
# ============================================================

class LichessBot:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING RL23 LICHESS BOT")
        print("=" * 60)

        # ----------------------------------------------------
        # Lichess client
        # ----------------------------------------------------

        self.client = LichessClient()

        # ----------------------------------------------------
        # RL23 engine
        # ----------------------------------------------------

        self.engine = ChessEngine(
            num_simulations=NUM_SIMULATIONS
        )

        # ----------------------------------------------------
        # Verify account
        # ----------------------------------------------------

        account = self.client.get_account()

        self.username = account["username"]
        self.title = account.get("title")

        print()
        print("Lichess username:", self.username)
        print("Account title:", self.title)

        if self.title != "BOT":
            raise RuntimeError(
                "The Lichess account is not a BOT account."
            )

        print()
        print("BOT account verified.")
        print("=" * 60)

    # ========================================================
    # RECONSTRUCT BOARD
    # ========================================================

    def create_board_from_moves(self, moves):

        board = chess.Board()

        if not moves:
            return board

        for move_uci in moves.split():

            try:

                move = chess.Move.from_uci(
                    move_uci
                )

            except ValueError:

                print(
                    "Invalid UCI move received:",
                    move_uci
                )

                continue

            if move not in board.legal_moves:

                print(
                    "WARNING: Illegal move received:",
                    move_uci
                )

                continue

            board.push(move)

        return board

    # ========================================================
    # MAKE ENGINE MOVE
    # ========================================================

    def make_engine_move(
        self,
        game_id,
        board
    ):

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if board.is_game_over():

            print("Game is already over.")

            return

        print()
        print("-" * 60)
        print("RL23 THINKING")
        print("-" * 60)

        print(board)

        print()
        print("FEN:")
        print(board.fen())

        print()
        print(
            f"MCTS simulations: "
            f"{NUM_SIMULATIONS}"
        )

        # ----------------------------------------------------
        # Think
        # ----------------------------------------------------

        start_time = time.time()

        move = self.engine.choose_move(board)

        elapsed = time.time() - start_time

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if move is None:

            print(
                "Engine returned no move."
            )

            return

        if move not in board.legal_moves:

            raise RuntimeError(
                f"ENGINE PRODUCED ILLEGAL MOVE: {move}"
            )

        move_uci = move.uci()

        print()
        print("=" * 60)
        print("RL23 MOVE")
        print("=" * 60)

        print("Move:", move)
        print("UCI:", move_uci)
        print(
            f"Thinking time: "
            f"{elapsed:.2f} seconds"
        )

        # ----------------------------------------------------
        # Send to Lichess
        # ----------------------------------------------------

        print()
        print("Sending move to Lichess...")

        self.client.make_move(
            game_id,
            move_uci
        )

        print("Move sent successfully.")

    # ========================================================
    # PLAY GAME
    # ========================================================

    def play_game(self, game_id):

        print()
        print("=" * 60)
        print("GAME STARTED")
        print("=" * 60)

        print("Game ID:", game_id)

        my_color = None

        # ----------------------------------------------------
        # Stream game
        # ----------------------------------------------------

        for event in self.client.stream_game(game_id):

            event_type = event.get("type")

            # =================================================
            # FULL GAME INFORMATION
            # =================================================

            if event_type == "gameFull":

                white = event.get(
                    "white",
                    {}
                )

                black = event.get(
                    "black",
                    {}
                )

                white_name = white.get(
                    "name",
                    ""
                )

                black_name = black.get(
                    "name",
                    ""
                )

                # ------------------------------------------------
                # Determine our color
                # ------------------------------------------------

                if (
                    white_name.lower()
                    == self.username.lower()
                ):

                    my_color = chess.WHITE

                elif (
                    black_name.lower()
                    == self.username.lower()
                ):

                    my_color = chess.BLACK

                else:

                    print(
                        "Could not determine bot color."
                    )

                    return

                print()
                print(
                    "RL23 is playing:",
                    "WHITE"
                    if my_color == chess.WHITE
                    else "BLACK"
                )

                # ------------------------------------------------
                # Current game state
                # ------------------------------------------------

                state = event.get(
                    "state",
                    {}
                )

                moves = state.get(
                    "moves",
                    ""
                )

                board = self.create_board_from_moves(
                    moves
                )

                # ------------------------------------------------
                # Check whose turn it is
                # ------------------------------------------------

                if board.turn == my_color:

                    self.make_engine_move(
                        game_id,
                        board
                    )

            # =================================================
            # GAME STATE UPDATE
            # =================================================

            elif event_type == "gameState":

                if my_color is None:
                    continue

                moves = event.get(
                    "moves",
                    ""
                )

                status = event.get(
                    "status"
                )

                # ------------------------------------------------
                # Check game status
                # ------------------------------------------------

                if status and status != "started":

                    print()
                    print("=" * 60)
                    print("GAME FINISHED")
                    print("=" * 60)

                    print("Status:", status)

                    return

                # ------------------------------------------------
                # Reconstruct board
                # ------------------------------------------------

                board = self.create_board_from_moves(
                    moves
                )

                # ------------------------------------------------
                # Check turn
                # ------------------------------------------------

                if board.turn == my_color:

                    self.make_engine_move(
                        game_id,
                        board
                    )

        print()
        print("Game stream ended.")

    # ========================================================
    # MAIN EVENT LOOP
    # ========================================================

    def run(self):

        print()
        print("=" * 60)
        print("RL23 LICHESS BOT ONLINE")
        print("=" * 60)

        print()
        print("Username:", self.username)
        print(
            "MCTS simulations:",
            NUM_SIMULATIONS
        )

        print()
        print("Waiting for Lichess events...")
        print()

        # ----------------------------------------------------
        # Event stream
        # ----------------------------------------------------

        for event in self.client.stream_events():

            event_type = event.get(
                "type"
            )

            # =================================================
            # CHALLENGE
            # =================================================

            if event_type == "challenge":

                challenge = event.get(
                    "challenge",
                    {}
                )

                challenge_id = challenge.get(
                    "id"
                )

                challenger = challenge.get(
                    "challenger",
                    {}
                )

                challenger_name = challenger.get(
                    "name",
                    "unknown"
                )

                print()
                print("=" * 60)
                print("NEW CHALLENGE")
                print("=" * 60)

                print(
                    "From:",
                    challenger_name
                )

                print(
                    "Challenge ID:",
                    challenge_id
                )

                # ------------------------------------------------
                # Accept
                # ------------------------------------------------

                try:

                    self.client.accept_challenge(
                        challenge_id
                    )

                    print(
                        "Challenge accepted."
                    )

                except Exception as e:

                    print(
                        "Could not accept challenge:"
                    )

                    print(e)

            # =================================================
            # GAME START
            # =================================================

            elif event_type == "gameStart":

                game = event.get(
                    "game",
                    {}
                )

                game_id = game.get(
                    "id"
                )

                if not game_id:
                    continue

                print()
                print(
                    "Game started:",
                    game_id
                )

                try:

                    self.play_game(
                        game_id
                    )

                except Exception as e:

                    print()
                    print("=" * 60)
                    print("GAME ERROR")
                    print("=" * 60)

                    print(e)

            # =================================================
            # GAME FINISH
            # =================================================

            elif event_type == "gameFinish":

                game = event.get(
                    "game",
                    {}
                )

                game_id = game.get(
                    "id"
                )

                print(
                    "Game finished:",
                    game_id
                )


# ============================================================
# START BOT
# ============================================================

if __name__ == "__main__":

    bot = LichessBot()

    bot.run()
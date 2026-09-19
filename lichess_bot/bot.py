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

BOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)


# ============================================================
# IMPORTS
# ============================================================

from engine import ChessEngine
from lichess_client import LichessClient


# ============================================================
# SETTINGS
# ============================================================

OPPONENT = "AetherBot"

NUM_SIMULATIONS = 100

CLOCK_LIMIT = 600
CLOCK_INCREMENT = 0


# ============================================================
# BOT
# ============================================================

class LichessBot:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING RL23 LICHESS BOT")
        print("=" * 60)

        # ----------------------------------------------------
        # Lichess
        # ----------------------------------------------------

        self.client = LichessClient()

        # ----------------------------------------------------
        # RL23
        # ----------------------------------------------------

        self.engine = ChessEngine(
            num_simulations=NUM_SIMULATIONS
        )

        # ----------------------------------------------------
        # Account
        # ----------------------------------------------------

        account = self.client.get_account()

        self.username = account["username"]
        self.title = account.get("title")

        print()
        print("Username:", self.username)
        print("Title:", self.title)

        if self.title != "BOT":
            raise RuntimeError(
                "Lichess account is not a BOT account."
            )

        print("BOT account verified.")

    # ========================================================
    # BOARD FROM MOVES
    # ========================================================

    def board_from_moves(self, moves):

        board = chess.Board()

        if not moves:
            return board

        for uci in moves.split():

            try:
                move = chess.Move.from_uci(uci)

            except ValueError:
                print(
                    "Invalid UCI received:",
                    uci
                )
                continue

            if move not in board.legal_moves:

                print(
                    "WARNING: illegal move:",
                    uci
                )

                continue

            board.push(move)

        return board

    # ========================================================
    # MAKE MOVE
    # ========================================================

    def make_move(self, game_id, board):

        if board.is_game_over():
            print("Game already finished.")
            return

        print()
        print("=" * 60)
        print("RL23 THINKING")
        print("=" * 60)

        print(board)

        print()
        print("FEN:")
        print(board.fen())

        start = time.time()

        move = self.engine.choose_move(board)

        elapsed = time.time() - start

        if move is None:
            print("Engine returned no move.")
            return

        if move not in board.legal_moves:
            raise RuntimeError(
                f"Illegal move from engine: {move}"
            )

        uci = move.uci()

        print()
        print("RL23 MOVE")
        print("-" * 60)
        print("Move:", move)
        print("UCI:", uci)
        print(
            f"Thinking time: {elapsed:.2f} seconds"
        )

        print()
        print("Sending move to Lichess...")

        self.client.make_move(
            game_id,
            uci
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

        for event in self.client.stream_game(game_id):

            event_type = event.get("type")

            print(
                "Game event:",
                event_type
            )

            # =================================================
            # FULL GAME
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
                    "RL23 color:",
                    "WHITE"
                    if my_color == chess.WHITE
                    else "BLACK"
                )

                state = event.get(
                    "state",
                    {}
                )

                moves = state.get(
                    "moves",
                    ""
                )

                board = self.board_from_moves(
                    moves
                )

                if board.turn == my_color:

                    self.make_move(
                        game_id,
                        board
                    )

            # =================================================
            # GAME STATE
            # =================================================

            elif event_type == "gameState":

                if my_color is None:
                    continue

                status = event.get(
                    "status"
                )

                # ------------------------------------------------
                # Game ended
                # ------------------------------------------------

                if status and status != "started":

                    print()
                    print("=" * 60)
                    print("GAME FINISHED")
                    print("=" * 60)

                    print("Status:", status)

                    return

                # ------------------------------------------------
                # Current moves
                # ------------------------------------------------

                moves = event.get(
                    "moves",
                    ""
                )

                board = self.board_from_moves(
                    moves
                )

                # ------------------------------------------------
                # Our turn
                # ------------------------------------------------

                if board.turn == my_color:

                    self.make_move(
                        game_id,
                        board
                    )

        print("Game stream ended.")

    # ========================================================
    # CREATE CHALLENGE
    # ========================================================

    def challenge_stockfish(self):

        print()
        print("=" * 60)
        print(f"CHALLENGING {OPPONENT}")
        print("=" * 60)

        result = self.client.challenge_user(
            username=OPPONENT,
            rated=False,
            clock_limit=CLOCK_LIMIT,
            clock_increment=CLOCK_INCREMENT,
            color="random"
        )

        challenge_id = result.get("id")

        print()
        print("Challenge created.")
        print("Challenge ID:", challenge_id)
        print("Opponent:", OPPONENT)
        print("Color:", result.get("finalColor"))
        print("Status:", result.get("status"))

        return challenge_id

    # ========================================================
    # EVENT LOOP
    # ========================================================

    def run(self):

        print()
        print("=" * 60)
        print("RL23 LICHESS BOT")
        print("=" * 60)

        # ----------------------------------------------------
        # Create challenge
        # ----------------------------------------------------

        challenge_id = self.challenge_stockfish()

        print()
        print(
            f"Waiting for {OPPONENT} to accept..."
        )

        print(
            "Challenge:",
            challenge_id
        )

        # ----------------------------------------------------
        # Listen for events
        # ----------------------------------------------------

        for event in self.client.stream_events():

            event_type = event.get(
                "type"
            )

            print(
                "Lichess event:",
                event_type
            )

            # =================================================
            # GAME START
            # =================================================

            if event_type == "gameStart":

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
                    "Game received:",
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

                    print(
                        repr(e)
                    )

                # ------------------------------------------------
                # After game ends, create another challenge?
                # ------------------------------------------------

                print()
                print(
                    "Game finished. "
                    "Bot is stopping."
                )

                return

            # =================================================
            # CHALLENGE CANCELLED
            # =================================================

            elif event_type == "challengeCanceled":

                challenge = event.get(
                    "challenge",
                    {}
                )

                if challenge.get("id") == challenge_id:

                    print(
                        "Challenge was cancelled."
                    )

                    return

            # =================================================
            # CHALLENGE DECLINED
            # =================================================

            elif event_type == "challengeDeclined":

                challenge = event.get(
                    "challenge",
                    {}
                )

                if challenge.get("id") == challenge_id:

                    print(
                        "Challenge was declined."
                    )

                    return


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    bot = LichessBot()

    bot.run()
import time
import chess

from engine import ChessEngine
from lichess_client import LichessClient


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SIMULATIONS = 100


# ============================================================
# LICHESS BOT
# ============================================================

class LichessBot:

    def __init__(self):
        print("=" * 60)
        print("INITIALIZING RL23 LICHESS BOT")
        print("=" * 60)

        # ----------------------------------------------------
        # Lichess API client
        # ----------------------------------------------------

        self.client = LichessClient()

        # ----------------------------------------------------
        # RL23 engine
        # ----------------------------------------------------

        self.engine = ChessEngine(
            num_simulations=NUM_SIMULATIONS
        )

        # ----------------------------------------------------
        # Account information
        # ----------------------------------------------------

        account = self.client.get_account()

        self.username = account["username"]

        print()
        print("Lichess account:", self.username)
        print("Account title:", account.get("title"))

        if account.get("title") != "BOT":
            raise RuntimeError(
                "This account is not a BOT account."
            )

        print()
        print("BOT account verified.")
        print("=" * 60)

    # ========================================================
    # PROCESS GAME
    # ========================================================

    def play_game(self, game_id):

        print()
        print("=" * 60)
        print("GAME STARTED")
        print("Game ID:", game_id)
        print("=" * 60)

        board = chess.Board()

        my_color = None
        game_started = False

        # ----------------------------------------------------
        # Receive game events
        # ----------------------------------------------------

        for event in self.client.stream_game(game_id):

            event_type = event.get("type")

            print()
            print("Game event:", event_type)

            # =================================================
            # GAME FULL
            # =================================================

            if event_type == "gameFull":

                white_player = event["white"].get("name", "")
                black_player = event["black"].get("name", "")

                if white_player.lower() == self.username.lower():
                    my_color = chess.WHITE

                elif black_player.lower() == self.username.lower():
                    my_color = chess.BLACK

                else:
                    print(
                        "Could not determine bot color."
                    )
                    return

                game_started = True

                print(
                    "Playing as:",
                    "WHITE" if my_color else "BLACK"
                )

                # ------------------------------------------------
                # Process moves already played
                # ------------------------------------------------

                state = event.get("state", {})

                moves = state.get("moves", "")

                if moves:
                    for move_uci in moves.split():
                        try:
                            move = chess.Move.from_uci(
                                move_uci
                            )

                            if move in board.legal_moves:
                                board.push(move)

                        except ValueError:
                            print(
                                "Invalid move:",
                                move_uci
                            )

                # ------------------------------------------------
                # Check if it is our turn
                # ------------------------------------------------

                if board.turn == my_color:
                    self.make_engine_move(
                        game_id,
                        board
                    )

            # =================================================
            # GAME STATE
            # =================================================

            elif event_type == "gameState":

                if not game_started:
                    continue

                moves = event.get("moves", "")

                # Reconstruct board from the complete move list.
                board = chess.Board()

                if moves:
                    for move_uci in moves.split():

                        try:
                            move = chess.Move.from_uci(
                                move_uci
                            )

                            if move in board.legal_moves:
                                board.push(move)

                            else:
                                print(
                                    "Illegal move received:",
                                    move_uci
                                )

                        except ValueError:
                            print(
                                "Invalid UCI move:",
                                move_uci
                            )

                # ------------------------------------------------
                # Game finished?
                # ------------------------------------------------

                status = event.get("status")

                if status and status != "started":

                    print()
                    print("=" * 60)
                    print("GAME FINISHED")
                    print("Status:", status)
                    print("=" * 60)

                    return

                # ------------------------------------------------
                # Our turn?
                # ------------------------------------------------

                if board.turn == my_color:

                    self.make_engine_move(
                        game_id,
                        board
                    )

        print("Game stream ended.")

    # ========================================================
    # ENGINE MOVE
    # ========================================================

    def make_engine_move(self, game_id, board):

        if board.is_game_over():

            print("Game is already over.")

            return

        print()
        print("-" * 60)
        print("RL23 THINKING")
        print("-" * 60)

        print(board)

        start_time = time.time()

        # ----------------------------------------------------
        # Ask RL23 engine for move
        # ----------------------------------------------------

        move = self.engine.choose_move(board)

        elapsed = time.time() - start_time

        if move is None:
            print("Engine returned no move.")
            return

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if move not in board.legal_moves:

            raise RuntimeError(
                f"Engine produced illegal move: {move}"
            )

        move_uci = move.uci()

        print()
        print("RL23 selected:", move_uci)
        print(f"Thinking time: {elapsed:.2f} seconds")

        # ----------------------------------------------------
        # Send move to Lichess
        # ----------------------------------------------------

        print("Sending move to Lichess...")

        self.client.make_move(
            game_id,
            move_uci
        )

        print("Move successfully sent.")

    # ========================================================
    # EVENT LOOP
    # ========================================================

    def run(self):

        print()
        print("=" * 60)
        print("RL23 LICHESS BOT ONLINE")
        print("=" * 60)

        print("Username:", self.username)
        print("MCTS simulations:", NUM_SIMULATIONS)

        print()
        print("Waiting for challenges/games...")
        print()

        # ----------------------------------------------------
        # Listen for Lichess events
        # ----------------------------------------------------

        for event in self.client.stream_events():

            event_type = event.get("type")

            print("Event:", event_type)

            # =================================================
            # NEW CHALLENGE
            # =================================================

            if event_type == "challenge":

                challenge = event.get("challenge", {})

                challenge_id = challenge.get("id")

                challenger = challenge.get(
                    "challenger",
                    {}
                )

                challenger_name = challenger.get(
                    "name",
                    "unknown"
                )

                print()
                print("New challenge!")
                print("From:", challenger_name)
                print("Challenge ID:", challenge_id)

                # ------------------------------------------------
                # Accept challenge
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
                        "Failed to accept challenge:",
                        e
                    )

            # =================================================
            # GAME START
            # =================================================

            elif event_type == "gameStart":

                game = event.get("game", {})

                game_id = game.get("id")

                if not game_id:
                    continue

                print()
                print("Starting game:", game_id)

                try:

                    self.play_game(game_id)

                except Exception as e:

                    print()
                    print(
                        "ERROR DURING GAME:"
                    )
                    print(e)

            # =================================================
            # GAME FINISH
            # =================================================

            elif event_type == "gameFinish":

                game = event.get("game", {})

                game_id = game.get("id")

                print(
                    "Game finished:",
                    game_id
                )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    bot = LichessBot()

    bot.run()
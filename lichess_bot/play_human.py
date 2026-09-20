
import chess
import time

from engine import ChessEngine
from lichess_client import LichessClient


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SIMULATIONS = 100


# ============================================================
# LICHESS RL30 BOT
# ============================================================

class RL30LichessBot:

    def __init__(self):

        print("=" * 60)
        print("RL30 LICHESS BOT")
        print("=" * 60)

        # ----------------------------------------------------
        # Lichess client
        # ----------------------------------------------------

        self.client = LichessClient()

        # ----------------------------------------------------
        # Account
        # ----------------------------------------------------

        self.account = self.client.get_account()

        self.username = self.account["username"]

        print(f"Lichess account: {self.username}")

        # ----------------------------------------------------
        # Engine
        # ----------------------------------------------------

        print("\nLoading RL30 engine...")

        self.engine = ChessEngine(
            num_simulations=NUM_SIMULATIONS
        )

        print("\nRL30 engine ready.")
        print("=" * 60)


    # ========================================================
    # ACCEPT CHALLENGE
    # ========================================================

    def handle_challenge(self, event):

        challenge = event.get("challenge")

        if not challenge:
            return

        challenge_id = challenge["id"]

        challenger = challenge["challenger"]["name"]

        variant = challenge.get("variant", {}).get(
            "key",
            "standard"
        )

        speed = challenge.get("speed", "")

        print("\n" + "=" * 60)
        print("NEW CHALLENGE")
        print("=" * 60)

        print(f"From:    {challenger}")
        print(f"Variant: {variant}")
        print(f"Speed:   {speed}")
        print(f"ID:      {challenge_id}")

        # ----------------------------------------------------
        # Only accept standard chess
        # ----------------------------------------------------

        if variant != "standard":

            print("Declining non-standard challenge.")

            self.client.decline_challenge(
                challenge_id
            )

            return

        # ----------------------------------------------------
        # Accept
        # ----------------------------------------------------

        print("Accepting challenge...")

        self.client.accept_challenge(
            challenge_id
        )

        print("Challenge accepted.")

        print("=" * 60)


    # ========================================================
    # BOARD FROM LICHESS MOVES
    # ========================================================

    def board_from_moves(self, moves):

        board = chess.Board()

        for move in moves:

            try:
                board.push_uci(move)

            except ValueError:

                print(
                    f"ERROR: Invalid move from Lichess: "
                    f"{move}"
                )

                raise

        return board


    # ========================================================
    # MAKE AI MOVE
    # ========================================================

    def make_ai_move(
        self,
        game_id,
        board
    ):

        print("\n" + "-" * 60)
        print("RL30 THINKING")
        print("-" * 60)

        print(board)

        start_time = time.time()

        move = self.engine.choose_move(board)

        thinking_time = time.time() - start_time

        if move is None:

            print("No move returned.")

            return

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if move not in board.legal_moves:

            raise RuntimeError(
                f"RL30 produced illegal move: {move}"
            )

        uci = move.uci()

        print(f"\nRL30 move: {uci}")
        print(
            f"Thinking time: "
            f"{thinking_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # Send to Lichess
        # ----------------------------------------------------

        print("Sending move to Lichess...")

        self.client.make_move(
            game_id,
            uci
        )

        print("Move sent.")


    # ========================================================
    # PLAY GAME
    # ========================================================

    def play_game(self, game_id):

        print("\n" + "=" * 60)
        print("GAME STARTED")
        print(f"Game ID: {game_id}")
        print("=" * 60)

        my_color = None

        for event in self.client.stream_game(game_id):

            # =================================================
            # GAME FULL
            # =================================================

            if event.get("type") == "gameFull":

                white = event["white"]["name"]
                black = event["black"]["name"]

                print(f"\nWhite: {white}")
                print(f"Black: {black}")

                if white.lower() == self.username.lower():

                    my_color = chess.WHITE

                elif black.lower() == self.username.lower():

                    my_color = chess.BLACK

                else:

                    print(
                        "ERROR: Could not determine bot color."
                    )

                    return

                print(
                    f"RL30 is playing: "
                    f"{'White' if my_color == chess.WHITE else 'Black'}"
                )

                state = event.get("state", {})

                moves = state.get("moves", "")

                move_list = (
                    moves.split()
                    if moves
                    else []
                )

                board = self.board_from_moves(
                    move_list
                )

                # ------------------------------------------------
                # If RL30 moves first
                # ------------------------------------------------

                if board.turn == my_color:

                    self.make_ai_move(
                        game_id,
                        board
                    )

            # =================================================
            # GAME STATE
            # =================================================

            elif event.get("type") == "gameState":

                moves = event.get(
                    "moves",
                    ""
                )

                move_list = (
                    moves.split()
                    if moves
                    else []
                )

                board = self.board_from_moves(
                    move_list
                )

                status = event.get(
                    "status",
                    "unknown"
                )

                # ------------------------------------------------
                # Game finished
                # ------------------------------------------------

                if status not in [
                    "started",
                    "created"
                ]:

                    print("\n" + "=" * 60)
                    print("GAME FINISHED")
                    print("=" * 60)

                    print(f"Status: {status}")
                    print(f"Moves: {moves}")

                    return

                # ------------------------------------------------
                # RL30 turn
                # ------------------------------------------------

                if board.turn == my_color:

                    self.make_ai_move(
                        game_id,
                        board
                    )

        print("\nGame stream ended.")


    # ========================================================
    # MAIN EVENT LOOP
    # ========================================================

    def run(self):

        print("\nWaiting for challenges...")
        print(
            "Challenge the RL30 bot from your "
            "test Lichess account."
        )

        for event in self.client.stream_events():

            event_type = event.get("type")

            # ------------------------------------------------
            # Incoming challenge
            # ------------------------------------------------

            if event_type == "challenge":

                self.handle_challenge(event)

            # ------------------------------------------------
            # Game started
            # ------------------------------------------------

            elif event_type == "gameStart":

                game_id = event["game"]["id"]

                print(
                    f"\nGame started: {game_id}"
                )

                self.play_game(game_id)

            # ------------------------------------------------
            # Game finished
            # ------------------------------------------------

            elif event_type == "gameFinish":

                game_id = event["game"]["id"]

                print(
                    f"\nGame finished: {game_id}"
                )

            # ------------------------------------------------
            # Other events
            # ------------------------------------------------

            else:

                print(
                    f"Event: {event_type}"
                )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        bot = RL30LichessBot()

        bot.run()

    except KeyboardInterrupt:

        print("\n\nBot stopped by user.")

    except Exception as e:

        print("\n" + "=" * 60)
        print("BOT ERROR")
        print("=" * 60)

        print(type(e).__name__)
        print(e)

        raise


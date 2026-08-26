import chess
import chess.pgn

import numpy as np

from environment.state_encoder import StateEncoder
from environment.action_encoder import ActionEncoder


class PGNDatasetBuilder:

    def __init__(self):

        self.action_encoder = ActionEncoder()

        self.samples = []


    def process_game(
        self,
        game
    ):

        board = game.board()


        # Determine game result

        result = game.headers.get(
            "Result",
            "*"
        )


        if result == "1-0":

            winner = chess.WHITE

        elif result == "0-1":

            winner = chess.BLACK

        elif result == "1/2-1/2":

            winner = None

        else:

            return


        for move in game.mainline_moves():

            # ------------------------------------------
            # Store position BEFORE the move
            # ------------------------------------------

            state = StateEncoder.encode(
                board
            )


            if move not in board.legal_moves:
                return

            try:
                action_id = self.action_encoder.encode(move)
            except KeyError:
                return


            policy = np.zeros(
                self.action_encoder.size(),
                dtype=np.float32
            )


            policy[action_id] = 1.0


            # ------------------------------------------
            # Value from current player's perspective
            # ------------------------------------------

            if winner is None:

                value = 0.0

            elif winner == board.turn:

                value = 1.0

            else:

                value = -1.0


            self.samples.append(

                (
                    state,
                    policy,
                    value
                )
            )


            # ------------------------------------------
            # Play move
            # ------------------------------------------

            board.push(move)


    def build_from_pgn(
        self,
        pgn_path,
        max_games=None
    ):

        games_processed = 0

        with open(
            pgn_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            while True:

                # ------------------------------------------
                # Game limit
                # ------------------------------------------

                if (
                    max_games is not None
                    and games_processed >= max_games
                ):
                    break


                # ------------------------------------------
                # Read next game
                # ------------------------------------------

                game = chess.pgn.read_game(
                    file
                )


                if game is None:
                    break


                # ------------------------------------------
                # Process game
                # ------------------------------------------

                self.process_game(
                    game
                )


                games_processed += 1


        print(
            "Games processed:",
            games_processed
        )


        return self.samples

        with open(
            pgn_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            while True:

                game = chess.pgn.read_game(
                    file
                )


                if game is None:

                    break


                self.process_game(
                    game
                )


        return self.samples
import chess


class ActionEncoder:

    PROMOTION_PIECES = [
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    ]

    def __init__(self):
        self.actions = []
        self.move_to_id = {}

        self._build_action_space()

    def _add_action(self, move):
        if move not in self.move_to_id:
            action_id = len(self.actions)

            self.actions.append(move)
            self.move_to_id[move] = action_id

    def _build_action_space(self):

        # Normal moves
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:

                if from_square == to_square:
                    continue

                move = chess.Move(from_square, to_square)

                self._add_action(move)

        # Promotion moves
        for from_square in chess.SQUARES:

            rank = chess.square_rank(from_square)

            # White promotion starts from rank 7
            if rank == 6:

                for to_square in chess.SQUARES:

                    if chess.square_rank(to_square) != 7:
                        continue

                    for promotion in self.PROMOTION_PIECES:

                        move = chess.Move(
                            from_square,
                            to_square,
                            promotion=promotion
                        )

                        self._add_action(move)

            # Black promotion starts from rank 2
            elif rank == 1:

                for to_square in chess.SQUARES:

                    if chess.square_rank(to_square) != 0:
                        continue

                    for promotion in self.PROMOTION_PIECES:

                        move = chess.Move(
                            from_square,
                            to_square,
                            promotion=promotion
                        )

                        self._add_action(move)

    def encode(self, move):
        return self.move_to_id[move]

    def decode(self, action_id):
        return self.actions[action_id]

    def size(self):
        return len(self.actions)
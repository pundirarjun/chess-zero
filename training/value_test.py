import chess


def value_for_position(
    board,
    winner
):

    if winner is None:

        return 0.0

    if winner == chess.WHITE:

        if board.turn == chess.WHITE:
            return 1.0

        return -1.0

    if winner == chess.BLACK:

        if board.turn == chess.BLACK:
            return 1.0

        return -1.0

    return 0.0
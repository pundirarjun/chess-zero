import numpy as np
import chess


class StateEncoder:

    PIECE_PLANES = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,

        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }

    @staticmethod
    def encode(board):
        state = np.zeros((18, 8, 8), dtype=np.float32)

        for square, piece in board.piece_map().items():
            plane = StateEncoder.PIECE_PLANES[
                (piece.piece_type, piece.color)
            ]

            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            state[plane, row, col] = 1.0

        if board.turn == chess.WHITE:
            state[12, :, :] = 1.0

        if board.has_kingside_castling_rights(chess.WHITE):
            state[13, :, :] = 1.0

        if board.has_queenside_castling_rights(chess.WHITE):
            state[14, :, :] = 1.0

        if board.has_kingside_castling_rights(chess.BLACK):
            state[15, :, :] = 1.0

        if board.has_queenside_castling_rights(chess.BLACK):
            state[16, :, :] = 1.0


        if board.ep_square is not None:
            row = 7 - chess.square_rank(board.ep_square)
            col = chess.square_file(board.ep_square)

            state[17, row, col] = 1.0

        return state
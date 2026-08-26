import chess
from environment.state_encoder import StateEncoder
from environment.action_encoder import ActionEncoder

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.action_encoder = ActionEncoder()

    def reset(self):
        self.board.reset()
        return self.get_state()

    def legal_moves(self):
        return list(self.board.legal_moves)

    def step(self, action):
        if action not in self.board.legal_moves:
            raise ValueError("Illegal move")

        self.board.push(action)

        done = self.board.is_game_over()

        if done:
            result = self.board.result()

            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            else:
                reward = 0
        else:
            reward = 0

        return self.get_state(), reward, done

    def get_state(self):
        return StateEncoder.encode(self.board)

    def legal_action_mask(self):
        mask = [0] * self.action_encoder.size()

        for move in self.board.legal_moves:
            action_id = self.action_encoder.encode(move)
            mask[action_id] = 1

        return mask
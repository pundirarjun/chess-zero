import chess
from mcts.policy import get_policy_for_board

class Node:

    def __init__(
        self,
        board,
        parent=None,
        move=None,
        prior=0.0
    ):

        self.board = board.copy()

        self.parent = parent

        self.move = move

        self.prior = prior

        self.visit_count = 0

        self.value_sum = 0.0

        self.children = {}

    @property
    def value(self):

        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def is_expanded(self):

        return len(self.children) > 0

    def is_terminal(self):

        return self.board.is_game_over(claim_draw=True)

    def expand(self, model, action_encoder):

        if self.is_terminal():
            return

        policy = get_policy_for_board(
            self.board,
            model,
            action_encoder
        )

        for move, prior in policy.items():

            child_board = self.board.copy()

            child_board.push(move)

            child = Node(
                board=child_board,
                parent=self,
                move=move,
                prior=prior
            )

            self.children[move] = child

    def puct_score(self, parent_visit_count, c_puct=1.5):

        exploration = (
            c_puct
            * self.prior
            * (parent_visit_count ** 0.5)
            / (1 + self.visit_count)
        )

        return -self.value + exploration

    def select_child(self, c_puct=1.5):

        best_move = None
        best_child = None
        best_score = float("-inf")

        for move, child in self.children.items():

            score = child.puct_score(
                self.visit_count,
                c_puct
            )

            if score > best_score:

                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def backup(self, value):

        node = self

        while node is not None:

            node.visit_count += 1

            node.value_sum += value

            value = -value

            node = node.parent
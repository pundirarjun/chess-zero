import chess

from mcts.policy import (
    get_policy_and_value_for_board
)


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

        # Real MCTS statistics
        self.visit_count = 0

        self.value_sum = 0.0

        # Temporary visits used only during
        # batched MCTS leaf selection
        self.virtual_visit_count = 0

        self.children = {}

    # ==================================================
    # VALUE
    # ==================================================

    @property
    def value(self):

        if self.visit_count == 0:
            return 0.0

        return (
            self.value_sum
            / self.visit_count
        )

    # ==================================================
    # EFFECTIVE VISITS
    # ==================================================

    @property
    def effective_visit_count(self):

        return (
            self.visit_count
            + self.virtual_visit_count
        )

    # ==================================================
    # TREE STATE
    # ==================================================

    def is_expanded(self):

        return len(self.children) > 0

    def is_terminal(self):

        return self.board.is_game_over(
            claim_draw=True
        )

    # ==================================================
    # NORMAL EXPANSION
    # ==================================================

    def expand(
        self,
        model,
        action_encoder
    ):

        if self.is_terminal():
            return None

        policy, value = (
            get_policy_and_value_for_board(
                self.board,
                model,
                action_encoder
            )
        )

        self.expand_with_policy(
            policy,
            action_encoder
        )

        return value

    # ==================================================
    # EXPANSION FROM ALREADY COMPUTED POLICY
    # ==================================================

    def expand_with_policy(
        self,
        policy,
        action_encoder
    ):

        if self.is_terminal():
            return

        # Prevent accidental double expansion
        if self.is_expanded():
            return

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

    # ==================================================
    # PUCT
    # ==================================================

    def puct_score(
        self,
        parent_visit_count,
        c_puct=1.5
    ):

        parent_visit_count = max(
            parent_visit_count,
            1
        )

        child_visit_count = (
            self.visit_count
            + self.virtual_visit_count
        )

        exploration = (
            c_puct
            * self.prior
            * (
                parent_visit_count ** 0.5
            )
            / (
                1 + child_visit_count
            )
        )

        return (
            -self.value
            + exploration
        )

    # ==================================================
    # SELECT CHILD
    # ==================================================

    def select_child(
        self,
        c_puct=1.5
    ):

        best_move = None

        best_child = None

        best_score = float("-inf")

        parent_visit_count = (
            self.visit_count
            + self.virtual_visit_count
        )

        for move, child in self.children.items():

            score = child.puct_score(
                parent_visit_count,
                c_puct
            )

            if score > best_score:

                best_score = score

                best_move = move

                best_child = child

        return (
            best_move,
            best_child
        )

    # ==================================================
    # BACKUP
    # ==================================================

    def backup(self, value):

        node = self

        while node is not None:

            node.visit_count += 1

            node.value_sum += value

            value = -value

            node = node.parent
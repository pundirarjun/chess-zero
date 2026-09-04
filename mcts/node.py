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

        # ==================================================
        # BOARD
        # ==================================================

        self.board = board.copy()

        # ==================================================
        # TREE RELATIONSHIPS
        # ==================================================

        self.parent = parent
        self.move = move
        self.prior = float(prior)

        # ==================================================
        # REAL MCTS STATISTICS
        # ==================================================

        self.visit_count = 0
        self.value_sum = 0.0

        # ==================================================
        # VIRTUAL VISITS
        #
        # Used only while selecting multiple leaves for a
        # neural-network batch.
        #
        # They are temporary and MUST be removed during
        # backup.
        # ==================================================

        self.virtual_visit_count = 0

        # ==================================================
        # CHILDREN
        # ==================================================

        self.children = {}

    # ======================================================
    # VALUE
    # ======================================================

    @property
    def value(self):

        if self.visit_count == 0:
            return 0.0

        return (
            self.value_sum
            / self.visit_count
        )

    # ======================================================
    # EFFECTIVE VISITS
    #
    # Real visits + temporary virtual visits.
    # ======================================================

    @property
    def effective_visit_count(self):

        return (
            self.visit_count
            + self.virtual_visit_count
        )

    # ======================================================
    # TREE STATE
    # ======================================================

    def is_expanded(self):

        return len(self.children) > 0

    def is_terminal(self):

        return self.board.is_game_over(
            claim_draw=True
        )

    # ======================================================
    # NORMAL EXPANSION
    #
    # Used by normal MCTS.
    #
    # Returns the neural-network value from the perspective
    # of the player to move in this node.
    # ======================================================

    def expand(
        self,
        model,
        action_encoder
    ):

        # Terminal positions cannot be expanded.

        if self.is_terminal():

            return None

        # Ask the neural network for:
        #
        # policy
        # value

        policy, value = (
            get_policy_and_value_for_board(
                self.board,
                model,
                action_encoder
            )
        )

        # Create child nodes.

        self.expand_with_policy(
            policy,
            action_encoder
        )

        return value

    # ======================================================
    # EXPANSION FROM ALREADY COMPUTED POLICY
    #
    # Used by batched MCTS after the neural network has
    # evaluated several positions simultaneously.
    # ======================================================

    def expand_with_policy(
        self,
        policy,
        action_encoder
    ):

        # Terminal nodes have no children.

        if self.is_terminal():

            return

        # Prevent accidental double expansion.

        if self.is_expanded():

            return

        # Create one child for every legal move contained
        # in the policy.

        for move, prior in policy.items():

            child_board = self.board.copy()

            child_board.push(move)

            child = Node(
                board=child_board,
                parent=self,
                move=move,
                prior=float(prior)
            )

            self.children[move] = child

    # ======================================================
    # PUCT SCORE
    #
    # Q = -child.value
    #
    # Because child.value is from the child's side-to-move
    # perspective, we negate it when evaluating the move
    # from the parent's perspective.
    # ======================================================

    def puct_score(
        self,
        parent_visit_count,
        c_puct=1.5
    ):

        # Numerical safety.

        parent_visit_count = max(
            parent_visit_count,
            1
        )

        # During batched MCTS, virtual visits temporarily
        # increase this count so that another simulation is
        # discouraged from selecting exactly the same path.

        child_visit_count = (
            self.visit_count
            + self.virtual_visit_count
        )

        # ==================================================
        # EXPLORATION TERM
        # ==================================================

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

        # ==================================================
        # PUCT
        # ==================================================

        return (
            -self.value
            + exploration
        )

    # ======================================================
    # SELECT CHILD
    # ======================================================

    def select_child(
        self,
        c_puct=1.5
    ):

        if not self.children:

            return None, None

        best_move = None
        best_child = None
        best_score = float("-inf")

        # Parent visits include virtual visits during
        # batched leaf selection.

        parent_visit_count = (
            self.visit_count
            + self.virtual_visit_count
        )

        # ==================================================
        # FIND CHILD WITH HIGHEST PUCT SCORE
        # ==================================================

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

    # ======================================================
    # BACKUP
    #
    # value:
    #   Value from the perspective of the node where the
    #   evaluation was originally produced.
    #
    # At every parent level the perspective changes, so
    # value is negated after each step.
    #
    # IMPORTANT:
    # Temporary virtual visits are removed here.
    # ======================================================

    def backup(self, value):

        node = self

        while node is not None:

            # ==================================================
            # REMOVE ONE TEMPORARY VIRTUAL VISIT
            #
            # A node can have virtual visits only because it
            # was included in one or more currently selected
            # batch paths.
            # ==================================================

            if node.virtual_visit_count > 0:

                node.virtual_visit_count -= 1

            # ==================================================
            # ADD REAL VISIT
            # ==================================================

            node.visit_count += 1

            # ==================================================
            # ADD VALUE
            # ==================================================

            node.value_sum += value

            # ==================================================
            # CHANGE PLAYER PERSPECTIVE
            # ==================================================

            value = -value

            # ==================================================
            # MOVE TO PARENT
            # ==================================================

            node = node.parent

    # ======================================================
    # RESET VIRTUAL VISITS
    #
    # Safety helper.
    #
    # Normally backup() should remove virtual visits
    # automatically. This method is useful if a batch is
    # interrupted or an exception occurs.
    # ======================================================

    def clear_virtual_visits(self):

        self.virtual_visit_count = 0

        for child in self.children.values():

            child.clear_virtual_visits()
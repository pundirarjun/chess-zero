from mcts.node import Node
from mcts.policy import get_value_for_board
import numpy as np

class MCTS:

    def __init__(
        self,
        model,
        action_encoder,
        c_puct=1.5
    ):

        self.model = model

        self.action_encoder = action_encoder

        self.c_puct = c_puct

    def run_simulation(self, root):

        node = root

        # -------------------------
        # Selection
        # -------------------------

        while node.is_expanded() and not node.is_terminal():

            _, node = node.select_child(
                self.c_puct
            )

        # -------------------------
        # Terminal position
        # -------------------------

        if node.is_terminal():

            value = self.get_terminal_value(node)

            node.backup(value)

            return

        # -------------------------
        # Expansion
        # -------------------------

        node.expand(
            self.model,
            self.action_encoder
        )

        # -------------------------
        # Evaluation
        # -------------------------

        value = get_value_for_board(
            node.board,
            self.model
        )

        # -------------------------
        # Backup
        # -------------------------

        node.backup(value)

    def get_terminal_value(self, node):

        outcome = node.board.outcome(
            claim_draw=True
        )

        if outcome is None:
            return 0.0

        if outcome.winner is None:
            return 0.0

        if outcome.winner == node.board.turn:
            return 1.0

        return -1.0

    def search(self, root, num_simulations):

        for _ in range(num_simulations):

            self.run_simulation(root)

    def select_action(self, root):

        if not root.children:
            raise ValueError("Root has not been searched.")

        best_move = None
        best_child = None
        best_visits = -1

        for move, child in root.children.items():

            if child.visit_count > best_visits:

                best_visits = child.visit_count
                best_move = move
                best_child = child

        return best_move, best_child

    def select_move(self, board, num_simulations=50):

        root = Node(board)

        self.search(
            root,
            num_simulations
        )

        # Sort moves by visit count
        sorted_children = sorted(
            root.children.items(),
            key=lambda item: item[1].visit_count,
            reverse=True
        )

        print("\nTop MCTS moves:")

        for move, child in sorted_children[:5]:

            print(
                move,
                "| visits:", child.visit_count,
                "| value:", round(child.value, 4),
                "| prior:", round(child.prior, 4)
            )

        move, _ = self.select_action(root)

        return move


    def get_policy_target(self, root):

        policy = [0.0] * self.action_encoder.size()

        total_visits = sum(
            child.visit_count
            for child in root.children.values()
        )

        if total_visits == 0:
            return policy

        for move, child in root.children.items():

            action_id = self.action_encoder.encode(move)

            policy[action_id] = (
                child.visit_count / total_visits
            )

        return policy


    def select_action_with_temperature(
        self,
        root,
        temperature=1.0
    ):

        if not root.children:
            raise ValueError(
                "Root has not been searched."
            )

        moves = list(root.children.keys())

        visits = np.array(
            [
                root.children[move].visit_count
                for move in moves
            ],
            dtype=np.float64
        )

        if temperature <= 0:

            best_index = np.argmax(visits)

            return moves[best_index]

        visits = visits ** (1.0 / temperature)

        probabilities = visits / visits.sum()

        selected_index = np.random.choice(
            len(moves),
            p=probabilities
        )

        return moves[selected_index]


    def add_dirichlet_noise(
        self,
        root,
        alpha=0.3,
        epsilon=0.25
    ):

        if not root.children:
            raise ValueError(
                "Root must be expanded before adding noise."
            )

        moves = list(root.children.keys())

        noise = np.random.dirichlet(
            [alpha] * len(moves)
        )

        for i, move in enumerate(moves):

            child = root.children[move]

            child.prior = (
                (1.0 - epsilon) * child.prior
                + epsilon * noise[i]
            )
import numpy as np
import torch

from environment.state_encoder import StateEncoder

from mcts.policy import policy_from_logits
from mcts.node import Node


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

    # ==================================================
    # NORMAL MCTS SIMULATION
    # ==================================================

    def run_simulation(self, root):

        node = root

        # --------------------------------------------------
        # Selection
        # --------------------------------------------------

        while (
            node.is_expanded()
            and not node.is_terminal()
        ):

            _, node = node.select_child(
                self.c_puct
            )

        # --------------------------------------------------
        # Terminal position
        # --------------------------------------------------

        if node.is_terminal():

            value = self.get_terminal_value(
                node
            )

            node.backup(value)

            return

        # --------------------------------------------------
        # Expansion + neural-network evaluation
        # --------------------------------------------------

        value = node.expand(
            self.model,
            self.action_encoder
        )

        # --------------------------------------------------
        # Backup
        # --------------------------------------------------

        node.backup(value)

    # ==================================================
    # TERMINAL VALUE
    # ==================================================

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

    # ==================================================
    # NORMAL SEARCH
    # ==================================================

    def search(
        self,
        root,
        num_simulations
    ):

        if num_simulations <= 0:

            return

        if root.is_terminal():

            return

        # --------------------------------------------------
        # Root initialization
        # --------------------------------------------------

        if not root.is_expanded():

            root.expand(
                self.model,
                self.action_encoder
            )

        # --------------------------------------------------
        # Actual simulations
        # --------------------------------------------------

        for _ in range(
            num_simulations
        ):

            self.run_simulation(
                root
            )

    # ==================================================
    # SELECT LEAF FOR BATCH
    # ==================================================

    def _select_leaf_for_batch(
        self,
        root,
        reserved=None
    ):

        if reserved is None:

            reserved = set()

        node = root

        path = [node]

        while (
            node.is_expanded()
            and not node.is_terminal()
        ):

            move, child = node.select_child(
                self.c_puct
            )

            if child is None:

                return None, path

            # --------------------------------------------------
            # Avoid selecting the same leaf twice in a batch.
            # --------------------------------------------------

            if id(child) in reserved:

                alternative_child = None
                alternative_move = None
                alternative_score = float("-inf")

                parent_visit_count = (
                    node.visit_count
                    + node.virtual_visit_count
                )

                for (
                    candidate_move,
                    candidate_child
                ) in node.children.items():

                    if id(candidate_child) in reserved:

                        continue

                    score = candidate_child.puct_score(
                        parent_visit_count,
                        self.c_puct
                    )

                    if score > alternative_score:

                        alternative_score = score
                        alternative_move = candidate_move
                        alternative_child = candidate_child

                if alternative_child is None:

                    return None, path

                child = alternative_child
                move = alternative_move

            node = child

            path.append(node)

        return node, path

    # ==================================================
    # EXPAND WITH POLICY
    # ==================================================

    def _expand_with_policy(
        self,
        node,
        policy
    ):

        node.expand_with_policy(
            policy,
            self.action_encoder
        )

    # ==================================================
    # BATCH NEURAL NETWORK EVALUATION
    # ==================================================

    def _evaluate_boards(
        self,
        boards
    ):

        if not boards:

            return None, None

        states = np.stack(
            [
                StateEncoder.encode(board)
                for board in boards
            ]
        ).astype(
            np.float32
        )

        device = next(
            self.model.parameters()
        ).device

        state_tensor = torch.from_numpy(
            states
        ).to(
            device
        )

        self.model.eval()

        with torch.no_grad():

            policy_logits, values = (
                self.model(
                    state_tensor
                )
            )

        return (
            policy_logits,
            values
        )

    # ==================================================
    # BATCHED MCTS
    # ==================================================

    def search_batched(
        self,
        root,
        num_simulations,
        batch_size=64
    ):

        if num_simulations <= 0:

            return

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be greater than 0."
            )

        if root.is_terminal():

            return

        # --------------------------------------------------
        # Root initialization
        # --------------------------------------------------

        if not root.is_expanded():

            root.expand(
                self.model,
                self.action_encoder
            )

        simulations_done = 0

        # ==================================================
        # BATCH LOOP
        # ==================================================

        while (
            simulations_done
            < num_simulations
        ):

            current_batch_size = min(
                batch_size,
                num_simulations - simulations_done
            )

            leaves = []
            paths = []

            reserved = set()

            # --------------------------------------------------
            # Select leaves
            # --------------------------------------------------

            for _ in range(
                current_batch_size
            ):

                leaf, path = (
                    self._select_leaf_for_batch(
                        root,
                        reserved
                    )
                )

                if leaf is None:

                    break

                for node in path:

                    node.virtual_visit_count += 1

                leaves.append(
                    leaf
                )

                paths.append(
                    path
                )

                reserved.add(
                    id(leaf)
                )

            if not leaves:

                break

            # ==================================================
            # TERMINAL / NON-TERMINAL SPLIT
            # ==================================================

            non_terminal_indices = []
            non_terminal_leaves = []

            for index, leaf in enumerate(
                leaves
            ):

                if leaf.is_terminal():

                    value = self.get_terminal_value(
                        leaf
                    )

                    leaf.backup(
                        value
                    )

                else:

                    non_terminal_indices.append(
                        index
                    )

                    non_terminal_leaves.append(
                        leaf
                    )

            # ==================================================
            # BATCH NEURAL NETWORK EVALUATION
            # ==================================================

            if non_terminal_leaves:

                policy_logits, values = (
                    self._evaluate_boards(
                        [
                            leaf.board
                            for leaf in non_terminal_leaves
                        ]
                    )
                )

                # --------------------------------------------------
                # Expand + backup
                # --------------------------------------------------

                for batch_index, original_index in enumerate(
                    non_terminal_indices
                ):

                    leaf = leaves[
                        original_index
                    ]

                    policy = policy_from_logits(
                        leaf.board,
                        policy_logits[
                            batch_index
                        ],
                        self.action_encoder
                    )

                    self._expand_with_policy(
                        leaf,
                        policy
                    )

                    value = values[
                        batch_index
                    ].item()

                    leaf.backup(
                        value
                    )

            simulations_done += len(
                leaves
            )

    # ==================================================
    # MULTI-GAME ROOT INITIALIZATION
    # ==================================================
    #
    # This evaluates multiple game roots in ONE GPU call.
    #
    # Root expansion itself is not counted as a simulation.
    #
    # ==================================================

    def _expand_roots_batched(
        self,
        roots
    ):

        pending_roots = [
            root
            for root in roots
            if (
                not root.is_terminal()
                and not root.is_expanded()
            )
        ]

        if not pending_roots:

            return

        policy_logits, _ = (
            self._evaluate_boards(
                [
                    root.board
                    for root in pending_roots
                ]
            )
        )

        for index, root in enumerate(
            pending_roots
        ):

            policy = policy_from_logits(
                root.board,
                policy_logits[index],
                self.action_encoder
            )

            self._expand_with_policy(
                root,
                policy
            )

    # ==================================================
    # MULTI-GAME BATCHED MCTS
    # ==================================================
    #
    # Runs MCTS for several independent game trees while
    # combining their neural-network evaluations into the
    # same GPU batches.
    #
    # Example:
    #
    # Game 1 ─┐
    # Game 2 ─┤
    # Game 3 ─┼──> GPU batch
    # Game 4 ─┤
    # Game 5 ─┘
    #
    # Each root receives exactly num_simulations actual
    # simulations.
    #
    # ==================================================

    def search_batched_multiple(
        self,
        roots,
        num_simulations,
        batch_size=128
    ):

        if not roots:

            return

        if num_simulations <= 0:

            return

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be greater than 0."
            )

        # --------------------------------------------------
        # Remove invalid roots.
        # --------------------------------------------------

        active_roots = [
            root
            for root in roots
            if not root.is_terminal()
        ]

        if not active_roots:

            return

        # --------------------------------------------------
        # Batch root initialization.
        # --------------------------------------------------

        self._expand_roots_batched(
            active_roots
        )

        # --------------------------------------------------
        # Track simulations separately for every game.
        # --------------------------------------------------

        simulations_done = {
            id(root): 0
            for root in active_roots
        }

        # ==================================================
        # MULTI-GAME BATCH LOOP
        # ==================================================

        while True:

            unfinished_roots = [
                root
                for root in active_roots
                if simulations_done[id(root)]
                < num_simulations
            ]

            if not unfinished_roots:

                break

            leaves = []
            paths = []
            leaf_roots = []

            reserved_by_root = {
                id(root): set()
                for root in unfinished_roots
            }

            # --------------------------------------------------
            # Fill one combined batch from multiple games.
            # --------------------------------------------------

            while (
                len(leaves) < batch_size
                and unfinished_roots
            ):

                made_progress = False

                for root in list(
                    unfinished_roots
                ):

                    if len(leaves) >= batch_size:

                        break

                    root_id = id(root)

                    if (
                        simulations_done[root_id]
                        >= num_simulations
                    ):

                        unfinished_roots.remove(
                            root
                        )

                        continue

                    leaf, path = (
                        self._select_leaf_for_batch(
                            root,
                            reserved_by_root[root_id]
                        )
                    )

                    if leaf is None:

                        simulations_done[root_id] = (
                            num_simulations
                        )

                        unfinished_roots.remove(
                            root
                        )

                        continue

                    for node in path:

                        node.virtual_visit_count += 1

                    leaves.append(
                        leaf
                    )

                    paths.append(
                        path
                    )

                    leaf_roots.append(
                        root
                    )

                    reserved_by_root[root_id].add(
                        id(leaf)
                    )

                    made_progress = True

                if not made_progress:

                    break

            if not leaves:

                break

            # ==================================================
            # TERMINAL / NON-TERMINAL SPLIT
            # ==================================================

            non_terminal_indices = []
            non_terminal_leaves = []

            for index, leaf in enumerate(
                leaves
            ):

                if leaf.is_terminal():

                    value = self.get_terminal_value(
                        leaf
                    )

                    leaf.backup(
                        value
                    )

                else:

                    non_terminal_indices.append(
                        index
                    )

                    non_terminal_leaves.append(
                        leaf
                    )

            # ==================================================
            # ONE COMBINED GPU EVALUATION
            # ==================================================

            if non_terminal_leaves:

                policy_logits, values = (
                    self._evaluate_boards(
                        [
                            leaf.board
                            for leaf in non_terminal_leaves
                        ]
                    )
                )

                # --------------------------------------------------
                # Expand + backup.
                # --------------------------------------------------

                for batch_index, original_index in enumerate(
                    non_terminal_indices
                ):

                    leaf = leaves[
                        original_index
                    ]

                    policy = policy_from_logits(
                        leaf.board,
                        policy_logits[
                            batch_index
                        ],
                        self.action_encoder
                    )

                    self._expand_with_policy(
                        leaf,
                        policy
                    )

                    value = values[
                        batch_index
                    ].item()

                    leaf.backup(
                        value
                    )

            # --------------------------------------------------
            # Every selected leaf represents one actual
            # simulation for its corresponding game.
            # --------------------------------------------------

            for root in leaf_roots:

                simulations_done[id(root)] += 1

    # ==================================================
    # SELECT ACTION
    # ==================================================

    def select_action(
        self,
        root
    ):

        if not root.children:

            raise ValueError(
                "Root has not been searched."
            )

        best_move = None
        best_child = None
        best_visits = -1

        for move, child in (
            root.children.items()
        ):

            if child.visit_count > best_visits:

                best_visits = (
                    child.visit_count
                )

                best_move = move
                best_child = child

        return (
            best_move,
            best_child
        )

    # ==================================================
    # SELECT MOVE
    # ==================================================

    def select_move(
        self,
        board,
        num_simulations=50
    ):

        root = Node(
            board
        )

        self.search(
            root,
            num_simulations
        )

        sorted_children = sorted(
            root.children.items(),
            key=lambda item:
                item[1].visit_count,
            reverse=True
        )

        print(
            "\nTop MCTS moves:"
        )

        for move, child in (
            sorted_children[:5]
        ):

            print(
                move,
                "| visits:",
                child.visit_count,
                "| value:",
                round(
                    child.value,
                    4
                ),
                "| prior:",
                round(
                    child.prior,
                    4
                )
            )

        move, _ = self.select_action(
            root
        )

        return move

    # ==================================================
    # POLICY TARGET
    # ==================================================

    def get_policy_target(
        self,
        root
    ):

        policy = [
            0.0
        ] * self.action_encoder.size()

        total_visits = sum(
            child.visit_count
            for child in root.children.values()
        )

        if total_visits == 0:

            return policy

        for move, child in (
            root.children.items()
        ):

            action_id = (
                self.action_encoder.encode(
                    move
                )
            )

            policy[action_id] = (
                child.visit_count
                / total_visits
            )

        return policy

    # ==================================================
    # TEMPERATURE ACTION
    # ==================================================

    def select_action_with_temperature(
        self,
        root,
        temperature=1.0
    ):

        if not root.children:

            raise ValueError(
                "Root has not been searched."
            )

        moves = list(
            root.children.keys()
        )

        visits = np.array(
            [
                root.children[move].visit_count
                for move in moves
            ],
            dtype=np.float64
        )

        if temperature <= 0:

            best_index = np.argmax(
                visits
            )

            return moves[
                best_index
            ]

        visits = (
            visits
            ** (1.0 / temperature)
        )

        total = visits.sum()

        if total <= 0:

            probabilities = np.ones(
                len(moves),
                dtype=np.float64
            )

            probabilities /= (
                probabilities.sum()
            )

        else:

            probabilities = (
                visits
                / total
            )

        selected_index = (
            np.random.choice(
                len(moves),
                p=probabilities
            )
        )

        return moves[
            selected_index
        ]

    # ==================================================
    # DIRICHLET NOISE
    # ==================================================

    def add_dirichlet_noise(
        self,
        root,
        alpha=0.3,
        epsilon=0.25
    ):

        if not root.children:

            raise ValueError(
                "Root must be expanded before "
                "adding noise."
            )

        if alpha <= 0:

            raise ValueError(
                "alpha must be greater than 0."
            )

        if not 0.0 <= epsilon <= 1.0:

            raise ValueError(
                "epsilon must be between 0 and 1."
            )

        moves = list(
            root.children.keys()
        )

        noise = np.random.dirichlet(
            [alpha] * len(moves)
        )

        for i, move in enumerate(
            moves
        ):

            child = root.children[
                move
            ]

            child.prior = (
                (1.0 - epsilon)
                * child.prior
                +
                epsilon
                * noise[i]
            )
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

        # Should normally not happen because the caller
        # already checked is_terminal(), but keep this safe.

        if outcome is None:

            return 0.0

        # Draw

        if outcome.winner is None:

            return 0.0

        # IMPORTANT:
        #
        # board.turn is the player to move.
        #
        # At a checkmate position, board.turn is the player
        # who has been checkmated.
        #
        # Therefore:
        #
        # winner == board.turn  -> +1
        # winner != board.turn  -> -1
        #
        # This keeps the value from the current node's
        # perspective.

        if outcome.winner == node.board.turn:

            return 1.0

        return -1.0

    # ==================================================
    # NORMAL SEARCH
    #
    # num_simulations means actual simulations.
    #
    # The root must be expanded before simulations begin.
    # If it is not expanded, expand it once first.
    #
    # Root expansion itself does NOT count as a simulation.
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

            value = root.expand(
                self.model,
                self.action_encoder
            )

            # Root evaluation is intentionally not backed up.
            #
            # The requested simulation count refers to actual
            # tree simulations.

        # --------------------------------------------------
        # Actual MCTS simulations
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

        # --------------------------------------------------
        # Traverse the tree using PUCT.
        #
        # Virtual visits are already included inside
        # Node.select_child().
        # --------------------------------------------------

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
            # If this child has already been selected as a leaf
            # for the current batch, try to avoid selecting it
            # again.
            #
            # Virtual visits normally make this unnecessary,
            # but the explicit check makes the behavior safer.
            # --------------------------------------------------

            if id(child) in reserved:

                alternative_child = None
                alternative_move = None
                alternative_score = float("-inf")

                parent_visit_count = (
                    node.visit_count
                    + node.virtual_visit_count
                )

                for candidate_move, candidate_child in (
                    node.children.items()
                ):

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
    #
    # Kept as a compatibility wrapper.
    #
    # Actual expansion is handled by Node.
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
    # BATCHED MCTS
    #
    # num_simulations = actual simulations.
    #
    # Root expansion does NOT consume a simulation.
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

        # --------------------------------------------------
        # Terminal root
        # --------------------------------------------------

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
        # Number of actual simulations completed
        # --------------------------------------------------

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

            # IDs of leaves already selected in this batch.

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

                # --------------------------------------------------
                # Apply one virtual visit to every node in
                # this selected path.
                #
                # This temporarily discourages another
                # simulation from following exactly the same path.
                # --------------------------------------------------

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

            # --------------------------------------------------
            # No leaves selected
            # --------------------------------------------------

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

                    # Terminal positions don't need neural
                    # network evaluation.

                    value = self.get_terminal_value(
                        leaf
                    )

                    # Node.backup() also removes one virtual
                    # visit from every node in this path.

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

                # --------------------------------------------------
                # Encode all leaf boards
                # --------------------------------------------------

                states = np.stack(
                    [
                        StateEncoder.encode(
                            leaf.board
                        )
                        for leaf in non_terminal_leaves
                    ]
                ).astype(
                    np.float32
                )

                # --------------------------------------------------
                # Get model device
                # --------------------------------------------------

                device = next(
                    self.model.parameters()
                ).device

                # --------------------------------------------------
                # Convert to tensor
                # --------------------------------------------------

                state_tensor = torch.from_numpy(
                    states
                ).to(
                    device
                )

                # --------------------------------------------------
                # Inference mode
                # --------------------------------------------------

                self.model.eval()

                with torch.no_grad():

                    policy_logits, values = (
                        self.model(
                            state_tensor
                        )
                    )

                # ==================================================
                # EXPAND + BACKUP EACH LEAF
                # ==================================================

                for batch_index, original_index in enumerate(
                    non_terminal_indices
                ):

                    leaf = leaves[
                        original_index
                    ]

                    # --------------------------------------------------
                    # Convert policy logits into legal-move policy
                    # --------------------------------------------------

                    policy = policy_from_logits(
                        leaf.board,
                        policy_logits[
                            batch_index
                        ],
                        self.action_encoder
                    )

                    # --------------------------------------------------
                    # Expand leaf
                    # --------------------------------------------------

                    self._expand_with_policy(
                        leaf,
                        policy
                    )

                    # --------------------------------------------------
                    # Neural-network value
                    # --------------------------------------------------

                    value = values[
                        batch_index
                    ].item()

                    # --------------------------------------------------
                    # Backup
                    #
                    # This ALSO removes one virtual visit from
                    # every node in the path.
                    # --------------------------------------------------

                    leaf.backup(
                        value
                    )

            # --------------------------------------------------
            # IMPORTANT
            #
            # DO NOT manually remove virtual visits here.
            #
            # Node.backup() already does that.
            # --------------------------------------------------

            simulations_done += len(
                leaves
            )

    # ==================================================
    # SELECT ACTION
    #
    # Deterministic:
    # choose the child with the highest visit count.
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
    #
    # Convenience method for playing a move using
    # normal MCTS.
    # ==================================================

    def select_move(
        self,
        board,
        num_simulations=50
    ):

        root = Node(
            board
        )

        # Root is initialized inside search().
        self.search(
            root,
            num_simulations
        )

        # --------------------------------------------------
        # Sort moves by visit count
        # --------------------------------------------------

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
    #
    # Converts root visit counts into a probability
    # distribution over the 4544-action space.
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
    #
    # temperature > 0:
    #   sample according to visit counts.
    #
    # temperature <= 0:
    #   choose highest visit count.
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

        # --------------------------------------------------
        # Deterministic selection
        # --------------------------------------------------

        if temperature <= 0:

            best_index = np.argmax(
                visits
            )

            return moves[
                best_index
            ]

        # --------------------------------------------------
        # Temperature scaling
        # --------------------------------------------------

        visits = (
            visits
            ** (1.0 / temperature)
        )

        total = visits.sum()

        # Safety fallback.

        if total <= 0:

            probabilities = np.ones(
                len(moves),
                dtype=np.float64
            )

            probabilities /= probabilities.sum()

        else:

            probabilities = (
                visits
                / total
            )

        # --------------------------------------------------
        # Sample action
        # --------------------------------------------------

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
    #
    # Used during self-play only.
    #
    # It should NOT be used during evaluation.
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
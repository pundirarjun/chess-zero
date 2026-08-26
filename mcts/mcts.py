import numpy as np
import torch

from environment.state_encoder import StateEncoder

from mcts.policy import (
    policy_from_logits
)

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
    # NORMAL SINGLE-SIMULATION MCTS
    # ==================================================

    def run_simulation(
        self,
        root
    ):

        node = root

        # ----------------------------------------------
        # Selection
        # ----------------------------------------------

        while (
            node.is_expanded()
            and not node.is_terminal()
        ):

            _, node = node.select_child(
                self.c_puct
            )

        # ----------------------------------------------
        # Terminal position
        # ----------------------------------------------

        if node.is_terminal():

            value = self.get_terminal_value(
                node
            )

            node.backup(value)

            return

        # ----------------------------------------------
        # Expansion + Evaluation
        # ----------------------------------------------

        value = node.expand(
            self.model,
            self.action_encoder
        )

        # ----------------------------------------------
        # Backup
        # ----------------------------------------------

        node.backup(value)

    # ==================================================
    # TERMINAL VALUE
    # ==================================================

    def get_terminal_value(
        self,
        node
    ):

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

        for _ in range(
            num_simulations
        ):

            self.run_simulation(
                root
            )

    # ==================================================
    # BATCH LEAF SELECTION
    # ==================================================

    def _select_leaf_for_batch(
        self,
        root
    ):

        node = root

        path = [node]

        # ----------------------------------------------
        # Follow PUCT path
        # ----------------------------------------------

        while (
            node.is_expanded()
            and not node.is_terminal()
        ):

            _, node = node.select_child(
                self.c_puct
            )

            path.append(node)

        return (
            node,
            path
        )

    # ==================================================
    # BATCHED SEARCH
    # ==================================================

def search_batched(
    self,
    root,
    num_simulations,
    batch_size=64
):
    """
    Batched MCTS search.

    Supports both:

    1. An unexpanded root
    2. A root that has already been expanded
       and optionally has Dirichlet noise applied

    Virtual visits are used during batch selection
    to prevent all leaves in a batch from following
    the same path.
    """

    if num_simulations <= 0:
        return

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

    simulations_done = 0

    # ==================================================
    # ROOT INITIALIZATION
    # ==================================================

    if root.is_terminal():
        return

    # Only expand/evaluate the root if it has not
    # already been expanded.
    #
    # This is important for self-play because
    # self-play expands the root first, applies
    # Dirichlet noise, and then calls this method.

    if not root.is_expanded():

        value = root.expand(
            self.model,
            self.action_encoder
        )

        root.backup(value)

        simulations_done += 1

    # ==================================================
    # BATCHED SEARCH
    # ==================================================

    while simulations_done < num_simulations:

        current_batch_size = min(
            batch_size,
            num_simulations - simulations_done
        )

        leaves = []
        paths = []

        # ----------------------------------------------
        # Select multiple leaves
        # ----------------------------------------------

        for _ in range(current_batch_size):

            leaf, path = (
                self._select_leaf_for_batch(
                    root
                )
            )

            if leaf is None:
                break

            # ------------------------------------------
            # Apply virtual visits to entire path
            # ------------------------------------------

            for node in path:

                node.virtual_visit_count += 1

            leaves.append(leaf)

            paths.append(path)

        if not leaves:
            break

        # ----------------------------------------------
        # Separate terminal leaves
        # ----------------------------------------------

        non_terminal_indices = []

        non_terminal_leaves = []

        for index, leaf in enumerate(leaves):

            if leaf.is_terminal():

                value = self.get_terminal_value(
                    leaf
                )

                leaf.backup(value)

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

            device = next(
                self.model.parameters()
            ).device

            state_tensor = (
                torch.from_numpy(
                    states
                ).to(device)
            )

            self.model.eval()

            with torch.no_grad():

                policy_logits, values = (
                    self.model(
                        state_tensor
                    )
                )

            # ==================================================
            # EXPAND + BACKUP
            # ==================================================

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

                leaf.expand_with_policy(
                    policy,
                    self.action_encoder
                )

                value = values[
                    batch_index
                ].item()

                leaf.backup(
                    value
                )

        # ==================================================
        # REMOVE VIRTUAL VISITS
        # ==================================================

        for path in paths:

            for node in path:

                if node.virtual_visit_count <= 0:

                    raise RuntimeError(
                        "Invalid virtual visit count."
                    )

                node.virtual_visit_count -= 1

        simulations_done += len(
            leaves
        )

    # ==================================================
    # MCTS POLICY TARGET
    # ==================================================

    def get_policy_target(
        self,
        root
    ):

        policy = [
            0.0
            for _ in range(
                self.action_encoder.size()
            )
        ]

        total_visits = sum(
            child.visit_count
            for child in
            root.children.values()
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
                root.children[
                    move
                ].visit_count
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

        probabilities = (
            visits
            / visits.sum()
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
                + epsilon
                * noise[i]
            )
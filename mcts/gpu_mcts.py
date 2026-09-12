"""Tensorized GPU Monte-Carlo Tree Search.

The tree, chess states, move generation, PUCT statistics, and neural-network
leaf evaluations all live on the selected torch device.  No python-chess board
or per-node Python object is used in the hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from environment.gpu_chess import GPUChess, MAX_LEGAL_MOVES


@dataclass
class GPUSearchResult:
    actions: torch.Tensor
    policy: torch.Tensor


class GPUMCTS:
    def __init__(
        self,
        model,
        action_encoder=None,
        device: Optional[torch.device | str] = None,
        c_puct: float = 1.5,
        max_children: int = MAX_LEGAL_MOVES,
    ):
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.action_encoder = action_encoder
        if self.action_encoder is not None and self.action_encoder.size() != 4544:
            raise ValueError("GPUMCTS requires the project's 4544-action space.")
        self.c_puct = float(c_puct)
        self.max_children = int(max_children)
        if self.max_children < 218:
            raise ValueError("max_children must be at least 218 for the full chess action space.")

        self.chess = GPUChess(self.device, 1)
        self._reset_tree()

    def _reset_tree(self):
        self.num_games = 0
        self.max_nodes = 0
        self.max_edges = 0
        self.pieces = None
        self.turn = None
        self.castling = None
        self.ep_square = None
        self.halfmove_clock = None
        self.fullmove_number = None
        self.parent = None
        self.visit_count = None
        self.value_sum = None
        self.expanded = None
        self.terminal = None
        self.edge_start = None
        self.edge_count = None
        self.edge_child = None
        self.edge_action = None
        self.edge_prior = None
        self.edge_used = 0

    def _allocate(self, num_games: int, num_simulations: int):
        # One root plus at most one newly expanded leaf per simulation and game.
        # Each expansion can have at most 218 legal chess moves, but 256 keeps
        # room for the complete 4544 action space's promotion structure.
        expansions = num_games * (num_simulations + 1)
        self.max_edges = expansions * self.max_children
        self.max_nodes = num_games + self.max_edges
        self.num_games = num_games

        dev = self.device
        self.pieces = torch.zeros((self.max_nodes, 12), dtype=torch.int64, device=dev)
        self.turn = torch.zeros((self.max_nodes,), dtype=torch.bool, device=dev)
        self.castling = torch.zeros((self.max_nodes,), dtype=torch.int16, device=dev)
        self.ep_square = torch.full((self.max_nodes,), -1, dtype=torch.int16, device=dev)
        self.halfmove_clock = torch.zeros((self.max_nodes,), dtype=torch.int16, device=dev)
        self.fullmove_number = torch.ones((self.max_nodes,), dtype=torch.int16, device=dev)

        self.parent = torch.full((self.max_nodes,), -1, dtype=torch.int32, device=dev)
        self.visit_count = torch.zeros((self.max_nodes,), dtype=torch.int32, device=dev)
        self.value_sum = torch.zeros((self.max_nodes,), dtype=torch.float32, device=dev)
        self.expanded = torch.zeros((self.max_nodes,), dtype=torch.bool, device=dev)
        self.terminal = torch.zeros((self.max_nodes,), dtype=torch.bool, device=dev)
        self.edge_start = torch.full((self.max_nodes,), -1, dtype=torch.int32, device=dev)
        self.edge_count = torch.zeros((self.max_nodes,), dtype=torch.int16, device=dev)

        self.edge_child = torch.full((self.max_edges,), -1, dtype=torch.int32, device=dev)
        self.edge_action = torch.zeros((self.max_edges,), dtype=torch.int16, device=dev)
        self.edge_prior = torch.zeros((self.max_edges,), dtype=torch.float32, device=dev)
        self.edge_used = 0

    def _state_view(self, node_ids: torch.Tensor) -> GPUChess:
        """Create a lightweight GPUChess view containing selected tree nodes."""
        g = object.__new__(GPUChess)
        g.__dict__ = self.chess.__dict__.copy()
        g.batch_size = int(node_ids.numel())
        g.pieces = self.pieces[node_ids]
        g.turn = self.turn[node_ids]
        g.castling = self.castling[node_ids]
        g.ep_square = self.ep_square[node_ids]
        g.halfmove_clock = self.halfmove_clock[node_ids]
        g.fullmove_number = self.fullmove_number[node_ids]
        return g

    def _write_states(self, node_ids: torch.Tensor, states: GPUChess):
        self.pieces[node_ids] = states.pieces
        self.turn[node_ids] = states.turn
        self.castling[node_ids] = states.castling
        self.ep_square[node_ids] = states.ep_square
        self.halfmove_clock[node_ids] = states.halfmove_clock
        self.fullmove_number[node_ids] = states.fullmove_number

    def _evaluate(self, node_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = self._state_view(node_ids)
        model_input = states.to_model_input()
        if model_input.is_cuda:
            model_input = model_input.contiguous(memory_format=torch.channels_last)
        self.model.eval()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, values = self.model(model_input)
            else:
                logits, values = self.model(model_input)
        legal = states.legal_move_mask()
        return logits.float(), values.squeeze(-1).float(), legal

    def _expand(self, node_ids: torch.Tensor, logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        if node_ids.numel() == 0:
            return torch.empty((0,), dtype=torch.bool, device=self.device)
        counts = legal.sum(dim=1).to(torch.int64)
        if bool((counts > self.max_children).any()):
            raise RuntimeError("A chess position exceeded the configured GPU MCTS child capacity.")

        total = int(counts.sum().item())
        if total == 0:
            self.terminal[node_ids] = True
            self.expanded[node_ids] = True
            return torch.ones((node_ids.numel(),), dtype=torch.bool, device=self.device)
        if self.edge_used + total > self.max_edges:
            raise RuntimeError("GPU MCTS edge pool exhausted; increase the search capacity.")
        if self.max_nodes < self.edge_used + total + self.num_games:
            raise RuntimeError("GPU MCTS node pool exhausted.")

        # Legal policy is computed entirely on GPU.  Invalid logits are masked
        # before softmax so no CPU-side move filtering is needed.
        masked = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        priors = torch.softmax(masked, dim=1)
        priors = priors * legal.to(priors.dtype)
        priors = priors / priors.sum(dim=1, keepdim=True).clamp_min(1e-12)

        pairs = torch.nonzero(legal, as_tuple=False)
        parent_local = pairs[:, 0]
        actions = pairs[:, 1].to(torch.long)
        parent_nodes = node_ids[parent_local]
        edge_start = self.edge_used
        edge_end = edge_start + total
        edge_ids = torch.arange(edge_start, edge_end, device=self.device, dtype=torch.long)

        # Child node IDs are one-to-one with edges.
        child_nodes = self.num_games + edge_ids
        if int(child_nodes[-1].item()) >= self.max_nodes:
            raise RuntimeError("GPU MCTS node pool exhausted.")

        self.parent[child_nodes] = parent_nodes.to(torch.int32)
        self.edge_child[edge_ids] = child_nodes.to(torch.int32)
        self.edge_action[edge_ids] = actions.to(torch.int16)
        self.edge_prior[edge_ids] = priors[parent_local, actions]

        # Because nonzero() is row-major, each parent occupies one contiguous
        # edge segment.  Prefix offsets recover those segments without Python
        # loops over individual children.
        prefix = torch.cumsum(counts, dim=0) - counts
        starts = edge_start + prefix
        self.edge_start[node_ids] = starts.to(torch.int32)
        self.edge_count[node_ids] = counts.to(torch.int16)

        # Apply all child actions as one GPU batch.
        parent_states = self._state_view(parent_nodes)
        child_states = parent_states.apply_actions_unchecked(
            torch.arange(total, device=self.device),
            actions,
        )
        self._write_states(child_nodes, child_states)

        # Draw conditions that do not require move generation can be marked
        # immediately.  Checkmate/stalemate are discovered when the child is
        # selected and its legal mask is evaluated; this avoids generating the
        # legal moves of every child during every expansion.
        child_view = self._state_view(child_nodes)
        draw_terminal = (child_view.halfmove_clock >= 100) | child_view.insufficient_material()
        self.terminal[child_nodes] = draw_terminal

        self.expanded[node_ids] = True
        self.edge_used = edge_end
        return torch.zeros((node_ids.numel(),), dtype=torch.bool, device=self.device)

    def _select_leaves(self, root_ids: torch.Tensor, max_depth: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Select one leaf per root in parallel and return leaf ids + paths."""
        g = root_ids.numel()
        current = root_ids.clone()
        active = torch.ones((g,), dtype=torch.bool, device=self.device)
        paths = torch.full((g, max_depth + 1), -1, dtype=torch.int32, device=self.device)
        paths[:, 0] = root_ids.to(torch.int32)
        depth = 0

        while depth < max_depth:
            selectable = active & self.expanded[current] & ~self.terminal[current] & (self.edge_count[current] > 0)
            if not bool(selectable.any()):
                break

            sel_rows = torch.nonzero(selectable, as_tuple=False).flatten()
            nodes = current[sel_rows]
            starts = self.edge_start[nodes].to(torch.long)
            counts = self.edge_count[nodes].to(torch.long)
            arange_child = torch.arange(self.max_children, device=self.device, dtype=torch.long)[None, :]
            edge_idx = starts[:, None] + arange_child
            valid = arange_child < counts[:, None]
            edge_idx_safe = edge_idx.clamp(0, self.max_edges - 1)
            child = self.edge_child[edge_idx_safe].to(torch.long)
            visits = self.visit_count[child].float()
            sums = self.value_sum[child]
            q = torch.where(visits > 0, sums / visits, torch.zeros_like(sums))
            priors = self.edge_prior[edge_idx_safe]
            parent_visits = self.visit_count[nodes].float().clamp_min(1.0)[:, None]
            scores = -q + self.c_puct * priors * torch.sqrt(parent_visits) / (1.0 + visits)
            scores = scores.masked_fill(~valid, -torch.inf)
            best = torch.argmax(scores, dim=1)
            next_nodes = child[torch.arange(sel_rows.numel(), device=self.device), best]
            current[sel_rows] = next_nodes
            depth += 1
            paths[:, depth] = paths[:, depth - 1]
            paths[sel_rows, depth] = next_nodes.to(torch.int32)

            # Rows that reached an unexpanded/terminal node stop. Others keep
            # traversing on the next depth iteration.
            active = active & self.expanded[current] & ~self.terminal[current]

        return current, paths[:, :depth + 1]

    def _backup(self, paths: torch.Tensor, values: torch.Tensor):
        # Back up each root's path from leaf to root.  Each row is independent;
        # duplicate-node collisions are possible only within a single tree and
        # are handled by the sequential depth updates.
        depth = paths.shape[1]
        v = values.clone()
        for d in range(depth - 1, -1, -1):
            nodes = paths[:, d].to(torch.long)
            valid = nodes >= 0
            if not bool(valid.any()):
                continue
            n = nodes[valid]
            vv = v[valid]
            self.visit_count.index_add_(0, n, torch.ones_like(vv, dtype=torch.int32))
            self.value_sum.index_add_(0, n, vv)
            v = -v

    def _terminal_values(self, node_ids: torch.Tensor) -> torch.Tensor:
        states = self._state_view(node_ids)
        legal = states.legal_move_mask()
        no_moves = ~legal.any(dim=1)
        check = states.is_in_check()
        checkmate = no_moves & check
        return torch.where(checkmate, -torch.ones_like(states.halfmove_clock, dtype=torch.float32), torch.zeros_like(states.halfmove_clock, dtype=torch.float32))

    def search(self,
    root_states: GPUChess,
    num_simulations: int,
    dirichlet_alpha: float | None = None,
    dirichlet_epsilon: float = 0.25,):
        if root_states.device != self.device:
            raise ValueError("root_states must live on the GPUMCTS device")
        if root_states.pieces.shape[0] <= 0:
            return
        if num_simulations < 0:
            raise ValueError("num_simulations must be non-negative")

        self._allocate(root_states.pieces.shape[0], max(1, num_simulations))
        roots = torch.arange(self.num_games, device=self.device, dtype=torch.long)
        self._write_states(roots, root_states)

        logits, _, legal = self._evaluate(roots)
        root_new_terminal = self._expand(roots, logits, legal)

        if dirichlet_alpha is not None:
            self.add_dirichlet_noise(
                dirichlet_alpha,
                dirichlet_epsilon,
            )

        if num_simulations <= 0:
            return

        # One simulation per root is selected at each round.  The resulting
        # leaf evaluations form one large GPU batch, maximizing network work.
        max_depth = max(8, num_simulations + 2)
        for _ in range(num_simulations):
            leaves, paths = self._select_leaves(roots, max_depth=max_depth)
            terminal = self.terminal[leaves]
            values = torch.zeros((self.num_games,), dtype=torch.float32, device=self.device)
            if bool(terminal.any()):
                t_ids = leaves[terminal]
                values[terminal] = self._terminal_values(t_ids)

            nonterminal = ~terminal
            if bool(nonterminal.any()):
                nt_ids = leaves[nonterminal]
                logits, nn_values, legal = self._evaluate(nt_ids)
                self._expand(nt_ids, logits, legal)
                no_moves = ~legal.any(dim=1)
                check = self._state_view(nt_ids).is_in_check()
                exact_terminal_value = torch.where(
                    no_moves & check,
                    -torch.ones_like(nn_values),
                    torch.zeros_like(nn_values),
                )
                newly_terminal = no_moves | (self._state_view(nt_ids).halfmove_clock >= 100) | self._state_view(nt_ids).insufficient_material()
                values[nonterminal] = torch.where(newly_terminal, exact_terminal_value, nn_values)

            self._backup(paths, values)

    def root_policy(self, temperature: float = 1.0) -> torch.Tensor:
        roots = torch.arange(self.num_games, device=self.device, dtype=torch.long)
        policy = torch.zeros((self.num_games, 4544), dtype=torch.float32, device=self.device)
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = torch.arange(self.max_children, device=self.device, dtype=torch.long)[None, :]
        edge_idx = starts[:, None] + cols
        valid = cols < counts[:, None]
        safe = edge_idx.clamp(0, self.max_edges - 1)
        actions = self.edge_action[safe].to(torch.long)
        visits = self.visit_count[self.edge_child[safe].to(torch.long)].float()
        if temperature <= 0:
            weights = visits
            best = torch.argmax(weights.masked_fill(~valid, -1), dim=1)
            chosen = actions[torch.arange(self.num_games, device=self.device), best]
            policy[torch.arange(self.num_games, device=self.device), chosen] = 1.0
            return policy
        weights = torch.where(valid, visits.clamp_min(0.0).pow(1.0 / temperature), torch.zeros_like(visits))
        denom = weights.sum(dim=1, keepdim=True)
        probs = weights / denom.clamp_min(1e-12)
        rows = torch.arange(self.num_games, device=self.device)[:, None].expand_as(actions)
        policy.scatter_add_(1, actions, torch.where(valid, probs, torch.zeros_like(probs)))
        return policy

    def select_actions(self, temperature: float = 1.0) -> torch.Tensor:
        roots = torch.arange(self.num_games, device=self.device, dtype=torch.long)
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = torch.arange(self.max_children, device=self.device, dtype=torch.long)[None, :]
        edge_idx = starts[:, None] + cols
        valid = cols < counts[:, None]
        safe = edge_idx.clamp(0, self.max_edges - 1)
        actions = self.edge_action[safe].to(torch.long)
        visits = self.visit_count[self.edge_child[safe].to(torch.long)].float()
        if temperature <= 0:
            return actions.gather(1, torch.argmax(visits.masked_fill(~valid, -1), dim=1, keepdim=True)).squeeze(1)
        weights = torch.where(valid, visits.clamp_min(0).pow(1.0 / temperature), torch.zeros_like(visits))
        probs = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return actions.gather(1, torch.multinomial(probs, 1)).squeeze(1)

    def add_dirichlet_noise(self, alpha: float = 0.3, epsilon: float = 0.25):
        if alpha <= 0 or not 0 <= epsilon <= 1:
            raise ValueError("Invalid Dirichlet parameters")
        roots = torch.arange(self.num_games, device=self.device, dtype=torch.long)
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        maxc = int(counts.max().item()) if counts.numel() else 0
        if maxc == 0:
            return
        cols = torch.arange(maxc, device=self.device, dtype=torch.long)[None, :]
        edge_idx = starts[:, None] + cols
        valid = cols < counts[:, None]
        # A Dirichlet distribution is sampled independently for each root.
        noise = torch.distributions.Dirichlet(torch.full((maxc,), alpha, device=self.device)).sample((self.num_games,))
        # Renormalize only over actual legal children.
        noise = noise * valid.float()
        noise = noise / noise.sum(dim=1, keepdim=True).clamp_min(1e-12)
        e = edge_idx.clamp(0, self.max_edges - 1)
        old = self.edge_prior[e]
        self.edge_prior[e] = torch.where(valid, (1 - epsilon) * old + epsilon * noise, old)


    def root_visit_policy(self) -> torch.Tensor:
        """Return normalized root visit counts over all 4544 actions."""
        roots = torch.arange(self.num_games, device=self.device, dtype=torch.long)
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = torch.arange(self.max_children, device=self.device, dtype=torch.long)[None, :]
        edge_idx = starts[:, None] + cols
        valid = cols < counts[:, None]
        safe = edge_idx.clamp(0, self.max_edges - 1)
        actions = self.edge_action[safe].to(torch.long)
        visits = self.visit_count[self.edge_child[safe].to(torch.long)].float()
        visits = torch.where(valid, visits, torch.zeros_like(visits))
        probs = visits / visits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        policy = torch.zeros((self.num_games, 4544), dtype=torch.float32, device=self.device)
        policy.scatter_add_(1, actions, probs)
        return policy

    def advance(self, actions: torch.Tensor) -> GPUChess:
        """Return the GPU states reached by one selected action per root."""
        actions = actions.to(device=self.device, dtype=torch.long)
        roots = torch.arange(self.num_games, device=self.device, dtype=torch.long)
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = torch.arange(self.max_children, device=self.device, dtype=torch.long)[None, :]
        edge_idx = starts[:, None] + cols
        valid = cols < counts[:, None]
        safe = edge_idx.clamp(0, self.max_edges - 1)
        edge_actions = self.edge_action[safe].to(torch.long)
        matches = (edge_actions == actions[:, None]) & valid
        found = matches.any(dim=1)
        if not bool(found.all()):
            raise RuntimeError("Selected GPU MCTS action is not a root child.")
        pos = torch.argmax(matches.to(torch.int8), dim=1)
        child_ids = self.edge_child[safe[torch.arange(self.num_games, device=self.device), pos]].to(torch.long)
        return self._state_view(child_ids)

    def get_child_states(self) -> GPUChess:
        roots = torch.arange(self.num_games, device=self.device, dtype=torch.long)
        actions = self.select_actions(temperature=0.0)
        root_starts = self.edge_start[roots].to(torch.long)
        root_counts = self.edge_count[roots].to(torch.long)
        cols = torch.arange(self.max_children, device=self.device, dtype=torch.long)[None, :]
        edge_idx = root_starts[:, None] + cols
        valid = cols < root_counts[:, None]
        safe = edge_idx.clamp(0, self.max_edges - 1)
        edge_actions = self.edge_action[safe].to(torch.long)
        chosen_pos = (edge_actions == actions[:, None]) & valid
        chosen_edge = torch.argmax(chosen_pos.to(torch.int8), dim=1)
        child_ids = self.edge_child[safe[torch.arange(self.num_games, device=self.device), chosen_edge]].to(torch.long)
        return self._state_view(child_ids)

    def root_children(self):
        roots = torch.arange(self.num_games, device=self.device, dtype=torch.long)
        starts = self.edge_start[roots].to(torch.long)
        counts = self.edge_count[roots].to(torch.long)
        cols = torch.arange(self.max_children, device=self.device, dtype=torch.long)[None, :]
        edge_idx = starts[:, None] + cols
        valid = cols < counts[:, None]
        safe = edge_idx.clamp(0, self.max_edges - 1)
        actions = self.edge_action[safe].to(torch.long)
        visits = self.visit_count[self.edge_child[safe].to(torch.long)].float()
        return actions, visits, valid

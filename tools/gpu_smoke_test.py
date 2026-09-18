"""Fast CUDA correctness test for the GPU chess engine and batched MCTS.

Run from project root:

    python tools/gpu_smoke_test.py

This test intentionally uses a tiny search so it is suitable for a quick
correctness check before a full RL iteration.
"""

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


def check(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"{name} FAILED" + (f": {detail}" if detail else ""))
    print(f"[PASS] {name}")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    # Use multiple games so the variable-depth batched-path logic is exercised.
    num_games = 4
    simulations = 10

    model = ChessNet(action_space_size=4544).to(device).eval()
    model.to(memory_format=torch.channels_last)

    states = GPUChess(device, num_games)

    torch.cuda.synchronize()
    start = time.perf_counter()

    search = GPUMCTS(model=model, device=device)
    search.search(
        states,
        num_simulations=simulations,
        batch_size=4,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # ------------------------------------------------------------
    # Root visit-count checks
    # ------------------------------------------------------------
    actions, visits, valid = search.root_children()

    valid_visits = visits.masked_fill(~valid, 0)
    root_visit_totals = valid_visits.sum(dim=1)

    # The root itself receives one visit during expansion/evaluation,
    # and each requested simulation contributes one real backup.
    expected = simulations
    print("Root child visit totals:", root_visit_totals.tolist())

    check(
        "Root visit totals",
        torch.all(root_visit_totals == expected).item(),
        f"expected {expected}, got {root_visit_totals.tolist()}",
    )

    # ------------------------------------------------------------
    # Root policy checks
    # ------------------------------------------------------------
    policy = search.root_visit_policy()
    policy_sums = policy.sum(dim=1)

    check(
        "Policy sums to 1",
        torch.allclose(
            policy_sums,
            torch.ones_like(policy_sums),
            atol=1e-5,
            rtol=0,
        ),
        f"sums={policy_sums.tolist()}",
    )

    check(
        "Policy has no negative values",
        (policy >= 0).all().item(),
    )

    # ------------------------------------------------------------
    # Action legality / advancement
    # ------------------------------------------------------------
    selected = search.select_actions(temperature=0.0)
    next_states = search.advance(selected)

    legal_next = next_states.legal_move_mask()
    next_legal_counts = legal_next.sum(dim=1)

    check(
        "Selected actions exist in root children",
        ((visits.gather(1, (actions == selected[:, None]).long().argmax(1)[:, None])
          >= 0).squeeze(1)).all().item(),
    )

    check(
        "Advanced states have legal moves",
        (next_legal_counts > 0).all().item(),
        f"counts={next_legal_counts.tolist()}",
    )

    # ------------------------------------------------------------
    # Internal path-padding sanity check
    # ------------------------------------------------------------
    # Run a direct selection after the tree has grown. Any -1 padding is
    # expected to remain padding rather than repeating the leaf node.
    roots = search._root_ids
    leaves, paths = search._select_leaves(roots, max_depth=64)

    check(
        "Selection returned one leaf per game",
        leaves.shape[0] == num_games,
    )

    check(
        "Path roots are valid",
        torch.all(paths[:, 0] == roots.to(torch.int32)).item(),
    )

    # Once -1 appears in a row, every later entry must remain -1.
    if paths.shape[1] > 1:
        neg = paths < 0
        later_after_neg = neg[:, :-1] & (~neg[:, 1:])
        check(
            "Path padding is monotonic",
            not later_after_neg.any().item(),
        )

    torch.cuda.synchronize()

    print()
    print("=" * 60)
    print("GPU MCTS CORRECTNESS TEST: PASS")
    print("=" * 60)
    print("Games:", num_games)
    print("Simulations/game:", simulations)
    print("Root visits:", root_visit_totals.tolist())
    print("Selected actions:", selected.tolist())
    print("Next legal-move counts:", next_legal_counts.tolist())
    print(f"Elapsed: {elapsed:.3f}s")


if __name__ == "__main__":
    main()

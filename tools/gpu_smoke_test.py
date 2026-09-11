"""Fast CUDA smoke test for the GPU chess engine and MCTS.

Run from project root:
    python tools/gpu_smoke_test.py
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


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    model = ChessNet(action_space_size=4544).to(device).eval()
    model.to(memory_format=torch.channels_last)
    states = GPUChess(device, 4)

    # First call includes CUDA context/kernel initialization.
    torch.cuda.synchronize()
    start = time.perf_counter()
    search = GPUMCTS(model=model, device=device)
    search.search(states, num_simulations=2)
    actions = search.select_actions(temperature=0.0)
    next_states = search.advance(actions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print("Games:", states.pieces.shape[0])
    print("Simulations/game:", 2)
    print("Selected actions:", actions.tolist())
    print("Next legal-move counts:", next_states.legal_move_mask().sum(1).tolist())
    print(f"Elapsed: {elapsed:.3f}s")
    print("GPU smoke test: PASS")


if __name__ == "__main__":
    main()

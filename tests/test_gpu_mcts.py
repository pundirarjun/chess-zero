import torch

from model.chess_net import ChessNet
from environment.gpu_chess import GPUChess
from mcts.gpu_mcts import GPUMCTS


def test_gpu_mcts_pipeline_cpu_validation():
    model = ChessNet(action_space_size=4544).eval()
    states = GPUChess("cpu", 2)
    search = GPUMCTS(model=model, device="cpu")
    search.search(states, num_simulations=2)
    policy = search.root_visit_policy()
    actions = search.select_actions(temperature=0.0)
    next_states = search.advance(actions)
    assert policy.shape == (2, 4544)
    assert torch.allclose(policy.sum(1), torch.ones(2), atol=1e-5)
    assert actions.shape == (2,)
    assert next_states.legal_move_mask().sum(1).min().item() > 0

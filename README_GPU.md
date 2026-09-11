# ChessAI Cloud — GPU-first build

This version keeps the existing neural-network architecture and 4544-action
space, but replaces the self-play hot path with a tensorized chess state and
GPU MCTS.

## Preserved interfaces

- 18 input planes, 8x8 board
- 128 hidden channels
- 8 residual blocks
- 4544 policy outputs
- scalar tanh value head
- existing RL checkpoint format
- RL iteration chain: Phase 1 -> RL1 -> RL2 -> ...

## GPU path

`training.self_play.play_games()` automatically uses `GPUMCTS` when the model
is on CUDA. The GPU path contains:

1. Tensorized chess board state and metadata
2. GPU pseudo-legal and king-safety filtering
3. GPU move application
4. GPU MCTS tree statistics and PUCT selection
5. Batched policy/value inference
6. GPU self-play state transitions
7. GPU FP16 training

Python-chess remains available for the PGN parser and legacy CPU compatibility,
but it is not used by the CUDA self-play hot path.

## Kaggle RL run

Edit only `RL_ITERATION` in `training/rl_training.py` for the next iteration.
For RL2 the script loads RL1 and writes RL2.

Recommended first diagnostic:

```text
NUM_SELF_PLAY_GAMES = 2
NUM_SIMULATIONS = 2
MAX_MOVES = 10
TRAINING_STEPS = 1
```

After the diagnostic succeeds, restore the production settings in the script.

## Tests

The tests validate the tensorized rules on CPU as a correctness oracle before
running CUDA. Run:

```bash
python -m pytest tests
```

or at minimum:

```bash
python -m compileall -q .
```

The GPU MCTS test uses CPU only as a deterministic CI/validation mode; when the
RL script runs on a CUDA model, the same implementation executes on CUDA.

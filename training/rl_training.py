import os
import torch

from model.chess_net import ChessNet
from environment.action_encoder import ActionEncoder
from training.self_play import play_game
from training.replay_buffer import ReplayBuffer
from training.dataset import ChessDataset
from training.trainer import train_one_batch


# ==================================================
# CONFIGURATION
# ==================================================

PREVIOUS_CHECKPOINT = (
    "checkpoints/rl_iteration_4.pt"
)

OUTPUT_CHECKPOINT = (
    "checkpoints/rl_iteration_5.pt"
)

# IMPORTANT:
# Use a NEW replay buffer for this experiment.
REPLAY_BUFFER_CHECKPOINT = (
    "checkpoints/replay_buffer_rl5.pt"
)

# Generate more games than the diagnostic.
NUM_SELF_PLAY_GAMES = 20

# Improved MCTS search.
NUM_SIMULATIONS = 100

MAX_MOVES = 200

# Self-play exploration.
TEMPERATURE = 1.0

TEMPERATURE_MOVES = 40

DIRICHLET_ALPHA = 0.3

DIRICHLET_EPSILON = 0.25

# Replay buffer.
REPLAY_BUFFER_CAPACITY = 10000

# RL training.
BATCH_SIZE = 8

TRAINING_STEPS = 50

LEARNING_RATE = 1e-4


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ==================================================
# MODEL
# ==================================================

action_encoder = ActionEncoder()

model = ChessNet(
    action_space_size=action_encoder.size()
)


# ==================================================
# LOAD PREVIOUS MODEL
# ==================================================

checkpoint = torch.load(
    PREVIOUS_CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()

print(
    "Loaded checkpoint:",
    PREVIOUS_CHECKPOINT
)


# ==================================================
# OPTIMIZER
# ==================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# FRESH REPLAY BUFFER
# ==================================================

replay_buffer = ReplayBuffer(
    capacity=REPLAY_BUFFER_CAPACITY
)


print(
    "\nStarting with a FRESH replay buffer."
)

print(
    "Old replay buffer will NOT be loaded."
)


# ==================================================
# SELF-PLAY
# ==================================================

print(
    "\n=============================="
)

print(
    "GENERATING SELF-PLAY GAMES"
)

print(
    "=============================="
)

print(
    "Games:",
    NUM_SELF_PLAY_GAMES
)

print(
    "MCTS simulations:",
    NUM_SIMULATIONS
)

print(
    "Temperature moves:",
    TEMPERATURE_MOVES
)

print(
    "Maximum moves:",
    MAX_MOVES
)


# ==================================================
# STATISTICS
# ==================================================

completed_games = 0

incomplete_games = 0

white_wins = 0

black_wins = 0

draws = 0

new_samples = 0

termination_counts = {}


# ==================================================
# GENERATE SELF-PLAY
# ==================================================

for game_number in range(
    1,
    NUM_SELF_PLAY_GAMES + 1
):

    print(
        f"\nGame {game_number}/"
        f"{NUM_SELF_PLAY_GAMES}"
    )

    result = play_game(
        model=model,
        num_simulations=NUM_SIMULATIONS,
        max_moves=MAX_MOVES,
        temperature=TEMPERATURE,
        temperature_moves=TEMPERATURE_MOVES,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON
    )


    # ----------------------------------------------
    # Print result
    # ----------------------------------------------

    print(
        "Completed:",
        result.completed
    )

    print(
        "Result:",
        result.result
    )

    print(
        "Termination:",
        result.termination
    )

    print(
        "Moves:",
        result.moves_played
    )


    # ----------------------------------------------
    # Termination statistics
    # ----------------------------------------------

    termination_counts[
        result.termination
    ] = termination_counts.get(
        result.termination,
        0
    ) + 1


    # ----------------------------------------------
    # Skip incomplete games
    # ----------------------------------------------

    if not result.completed:

        incomplete_games += 1

        print(
            "Skipping incomplete game."
        )

        continue


    # ----------------------------------------------
    # Add completed game
    # ----------------------------------------------

    replay_buffer.add(
        result.training_data
    )

    completed_games += 1

    new_samples += len(
        result.training_data
    )


    # ----------------------------------------------
    # Result statistics
    # ----------------------------------------------

    if result.result == 1:

        white_wins += 1

    elif result.result == -1:

        black_wins += 1

    elif result.result == 0:

        draws += 1


    print(
        "Training samples added:",
        len(result.training_data)
    )


# ==================================================
# SELF-PLAY SUMMARY
# ==================================================

print(
    "\n=============================="
)

print(
    "SELF-PLAY SUMMARY"
)

print(
    "=============================="
)

print(
    "Completed games:",
    completed_games
)

print(
    "Incomplete games:",
    incomplete_games
)

print(
    "White wins:",
    white_wins
)

print(
    "Black wins:",
    black_wins
)

print(
    "Draws:",
    draws
)

print(
    "New samples:",
    new_samples
)

print(
    "Replay buffer size:",
    len(replay_buffer)
)


# ==================================================
# GAME RESULT DISTRIBUTION
# ==================================================

if completed_games > 0:

    print(
        "\nCompleted-game distribution:"
    )

    print(
        "White win:",
        f"{white_wins / completed_games:.2%}"
    )

    print(
        "Black win:",
        f"{black_wins / completed_games:.2%}"
    )

    print(
        "Draw:",
        f"{draws / completed_games:.2%}"
    )


# ==================================================
# TERMINATION REASONS
# ==================================================

print(
    "\nTermination reasons:"
)

for reason, count in termination_counts.items():

    print(
        reason,
        ":",
        count
    )


# ==================================================
# CHECK REPLAY BUFFER
# ==================================================

if len(replay_buffer) == 0:

    raise RuntimeError(
        "No completed self-play samples "
        "were generated."
    )


# ==================================================
# SAVE FRESH REPLAY BUFFER
# ==================================================

torch.save(
    list(replay_buffer),
    REPLAY_BUFFER_CHECKPOINT
)

print(
    "\nReplay buffer saved:"
)

print(
    REPLAY_BUFFER_CHECKPOINT
)


# ==================================================
# RL TRAINING
# ==================================================

print(
    "\n=============================="
)

print(
    "RL TRAINING"
)

print(
    "=============================="
)

print(
    "Training samples:",
    len(replay_buffer)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Training steps:",
    TRAINING_STEPS
)

print(
    "Learning rate:",
    LEARNING_RATE
)


# ==================================================
# TRAINING LOOP
# ==================================================

step = 0

total_loss_sum = 0.0

policy_loss_sum = 0.0

value_loss_sum = 0.0


while step < TRAINING_STEPS:

    # ----------------------------------------------
    # Sample random batch
    # ----------------------------------------------

    batch_samples = replay_buffer.sample(
        BATCH_SIZE
    )


    # ----------------------------------------------
    # Create dataset
    # ----------------------------------------------

    dataset = ChessDataset(
        batch_samples
    )


    states = []

    policies = []

    values = []


    for state, policy, value in dataset:

        states.append(
            state
        )

        policies.append(
            policy
        )

        values.append(
            value
        )


    states = torch.stack(
        states
    ).to(device)


    policies = torch.stack(
        policies
    ).to(device)


    values = torch.stack(
        values
    ).to(device)


    # ----------------------------------------------
    # Training step
    # ----------------------------------------------

    total_loss, policy_loss, value_loss = (
        train_one_batch(
            model=model,
            optimizer=optimizer,
            states=states,
            target_policy=policies,
            target_value=values
        )
    )


    step += 1


    # ----------------------------------------------
    # Accumulate statistics
    # ----------------------------------------------

    total_loss_sum += total_loss

    policy_loss_sum += policy_loss

    value_loss_sum += value_loss


    # ----------------------------------------------
    # Print progress
    # ----------------------------------------------

    print(
        f"Step {step}: "
        f"Total={total_loss:.6f} | "
        f"Policy={policy_loss:.6f} | "
        f"Value={value_loss:.6f}"
    )


# ==================================================
# TRAINING SUMMARY
# ==================================================

print(
    "\n=============================="
)

print(
    "TRAINING SUMMARY"
)

print(
    "=============================="
)

print(
    "Average total loss:",
    f"{total_loss_sum / TRAINING_STEPS:.6f}"
)

print(
    "Average policy loss:",
    f"{policy_loss_sum / TRAINING_STEPS:.6f}"
)

print(
    "Average value loss:",
    f"{value_loss_sum / TRAINING_STEPS:.6f}"
)


# ==================================================
# SAVE RL ITERATION 5
# ==================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "iteration":
            5,

        "previous_checkpoint":
            PREVIOUS_CHECKPOINT,

        "replay_buffer_size":
            len(replay_buffer),

        "completed_games":
            completed_games,

        "incomplete_games":
            incomplete_games,

        "new_samples":
            new_samples,

        "white_wins":
            white_wins,

        "black_wins":
            black_wins,

        "draws":
            draws,

        "num_simulations":
            NUM_SIMULATIONS,

        "temperature_moves":
            TEMPERATURE_MOVES
    },
    OUTPUT_CHECKPOINT
)


# ==================================================
# COMPLETE
# ==================================================

print(
    "\n=============================="
)

print(
    "RL ITERATION 5 COMPLETE"
)

print(
    "=============================="
)

print(
    "Checkpoint saved:"
)

print(
    OUTPUT_CHECKPOINT
)

print(
    "\nFresh replay buffer:"
)

print(
    REPLAY_BUFFER_CHECKPOINT
)
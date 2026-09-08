import os
import sys
import torch


# ==========================================================
# MAKE PROJECT ROOT IMPORTABLE
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# IMPORTS
# ==========================================================

from model.chess_net import ChessNet
from environment.action_encoder import ActionEncoder
from training.self_play import play_game
from training.replay_buffer import ReplayBuffer
from training.trainer import train_one_batch


# ==========================================================
# CONFIGURATION
# ==========================================================

# RL Iteration 1 model
PREVIOUS_CHECKPOINT = (
    "/kaggle/input/datasets/arjunthakur9999/checkpoints/rl_iteration_1.pt"
)

# RL Iteration 2 output
OUTPUT_CHECKPOINT = (
    "checkpoints/rl_iteration_2.pt"
)

# RL Iteration 2 replay buffer
REPLAY_BUFFER_CHECKPOINT = (
    "checkpoints/replay_buffer_rl2.pt"
)


# ==========================================================
# SELF-PLAY CONFIGURATION
# ==========================================================

NUM_SELF_PLAY_GAMES = 100

NUM_SIMULATIONS = 100

MAX_MOVES = 300

MCTS_BATCH_SIZE = 64

TEMPERATURE = 1.0

TEMPERATURE_MOVES = 40

DIRICHLET_ALPHA = 0.3

DIRICHLET_EPSILON = 0.25


# ==========================================================
# REPLAY BUFFER
# ==========================================================

REPLAY_BUFFER_CAPACITY = 50000


# ==========================================================
# RL TRAINING
# ==========================================================

TRAINING_BATCH_SIZE = 32

TRAINING_STEPS = 200

LEARNING_RATE = 1e-4


# ==========================================================
# ITERATION
# ==========================================================

ITERATION = 2


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# CREATE MODEL
# ==========================================================

def create_model():

    action_encoder = ActionEncoder()

    model = ChessNet(
        action_space_size=action_encoder.size()
    )

    model.to(device)

    return model, action_encoder


# ==========================================================
# CREATE OPTIMIZER
# ==========================================================

def create_optimizer(model):

    return torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


# ==========================================================
# LOAD PREVIOUS RL CHECKPOINT
# ==========================================================

def load_previous_checkpoint(
    model,
    optimizer,
    checkpoint_path
):

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Starting checkpoint not found: "
            f"{checkpoint_path}\n"
            f"Run RL Iteration 1 first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # RL starts with a fresh optimizer.
    for param_group in optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    model.to(device)
    model.train()

    return checkpoint


# ==========================================================
# GENERATE SELF-PLAY DATA
# ==========================================================

def generate_self_play_data(
    model,
    replay_buffer
):

    print(
        "\n=============================="
    )

    print(
        "GENERATING SELF-PLAY GAMES"
    )

    print(
        "=============================="
    )

    completed_games = 0
    incomplete_games = 0

    white_wins = 0
    black_wins = 0
    draws = 0

    new_samples = 0

    termination_counts = {}

    for game_number in range(
        1,
        NUM_SELF_PLAY_GAMES + 1
    ):

        print(
            f"\n========== GAME "
            f"{game_number}/{NUM_SELF_PLAY_GAMES} =========="
        )

        result = play_game(
            model=model,
            num_simulations=NUM_SIMULATIONS,
            max_moves=MAX_MOVES,
            temperature=TEMPERATURE,
            temperature_moves=TEMPERATURE_MOVES,
            dirichlet_alpha=DIRICHLET_ALPHA,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            batch_size=MCTS_BATCH_SIZE
        )

        termination_counts[
            result.termination
        ] = termination_counts.get(
            result.termination,
            0
        ) + 1

        # --------------------------------------------------
        # Discard incomplete games
        # --------------------------------------------------

        if not result.completed:

            incomplete_games += 1

            print(
                "Skipping incomplete game."
            )

            continue

        completed_games += 1

        # --------------------------------------------------
        # Add training data
        # --------------------------------------------------

        if result.training_data:

            replay_buffer.add(
                result.training_data
            )

            new_samples += len(
                result.training_data
            )

        # --------------------------------------------------
        # Result statistics
        # --------------------------------------------------

        if result.result == 1:

            white_wins += 1

        elif result.result == -1:

            black_wins += 1

        elif result.result == 0:

            draws += 1

    # ======================================================
    # SUMMARY
    # ======================================================

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

    print(
        "\nTermination reasons:"
    )

    for reason, count in termination_counts.items():

        print(
            reason,
            ":",
            count
        )

    return {
        "completed_games": completed_games,
        "incomplete_games": incomplete_games,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "new_samples": new_samples,
        "termination_counts": termination_counts,
        "replay_buffer_size": len(replay_buffer)
    }


# ==========================================================
# SAVE REPLAY BUFFER
# ==========================================================

def save_replay_buffer(
    replay_buffer,
    path
):

    directory = os.path.dirname(path)

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    torch.save(
        list(replay_buffer),
        path
    )

    print(
        "\nReplay buffer saved:",
        path
    )


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(
    model,
    optimizer,
    replay_buffer
):

    if len(replay_buffer) < TRAINING_BATCH_SIZE:

        raise RuntimeError(
            f"Replay buffer contains "
            f"{len(replay_buffer)} samples, "
            f"but training batch size is "
            f"{TRAINING_BATCH_SIZE}."
        )

    print(
        "\n=============================="
    )

    print(
        "RL TRAINING"
    )

    print(
        "=============================="
    )

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    for step in range(
        1,
        TRAINING_STEPS + 1
    ):

        batch_samples = replay_buffer.sample(
            TRAINING_BATCH_SIZE
        )

        states = torch.stack(
            [
                torch.from_numpy(
                    sample[0]
                ).float()
                for sample in batch_samples
            ]
        ).to(device)

        policies = torch.stack(
            [
                torch.from_numpy(
                    sample[1]
                ).float()
                for sample in batch_samples
            ]
        ).to(device)

        values = torch.stack(
            [
                torch.tensor(
                    sample[2],
                    dtype=torch.float32
                )
                for sample in batch_samples
            ]
        ).to(device)

        total_loss, policy_loss, value_loss = train_one_batch(
            model=model,
            optimizer=optimizer,
            states=states,
            target_policy=policies,
            target_value=values
        )

        total_loss_sum += total_loss
        policy_loss_sum += policy_loss
        value_loss_sum += value_loss

        print(
            f"Step {step}: "
            f"Total={total_loss:.6f} | "
            f"Policy={policy_loss:.6f} | "
            f"Value={value_loss:.6f}"
        )

    return {
        "average_total_loss":
            total_loss_sum / TRAINING_STEPS,

        "average_policy_loss":
            policy_loss_sum / TRAINING_STEPS,

        "average_value_loss":
            value_loss_sum / TRAINING_STEPS
    }


# ==========================================================
# SAVE RL CHECKPOINT
# ==========================================================

def save_rl_checkpoint(
    model,
    optimizer,
    self_play_stats,
    training_stats
):

    directory = os.path.dirname(
        OUTPUT_CHECKPOINT
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "iteration":
            ITERATION,

        "previous_checkpoint":
            PREVIOUS_CHECKPOINT,

        "completed_games":
            self_play_stats["completed_games"],

        "incomplete_games":
            self_play_stats["incomplete_games"],

        "white_wins":
            self_play_stats["white_wins"],

        "black_wins":
            self_play_stats["black_wins"],

        "draws":
            self_play_stats["draws"],

        "new_samples":
            self_play_stats["new_samples"],

        "replay_buffer_size":
            self_play_stats["replay_buffer_size"],

        "num_simulations":
            NUM_SIMULATIONS,

        "mcts_batch_size":
            MCTS_BATCH_SIZE,

        "temperature":
            TEMPERATURE,

        "temperature_moves":
            TEMPERATURE_MOVES,

        "dirichlet_alpha":
            DIRICHLET_ALPHA,

        "dirichlet_epsilon":
            DIRICHLET_EPSILON,

        "max_moves":
            MAX_MOVES,

        "training_batch_size":
            TRAINING_BATCH_SIZE,

        "training_steps":
            TRAINING_STEPS,

        "learning_rate":
            LEARNING_RATE,

        "average_total_loss":
            training_stats["average_total_loss"],

        "average_policy_loss":
            training_stats["average_policy_loss"],

        "average_value_loss":
            training_stats["average_value_loss"]
    }

    torch.save(
        checkpoint,
        OUTPUT_CHECKPOINT
    )

    print(
        "\nRL checkpoint saved:",
        OUTPUT_CHECKPOINT
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "=========================================="
    )

    print(
        "RL TRAINING"
    )

    print(
        "=========================================="
    )

    print(
        "Iteration:",
        ITERATION
    )

    print(
        "Device:",
        device
    )

    print(
        "Previous checkpoint:",
        PREVIOUS_CHECKPOINT
    )

    print(
        "Output checkpoint:",
        OUTPUT_CHECKPOINT
    )

    # ------------------------------------------------------
    # Create model
    # ------------------------------------------------------

    model, action_encoder = create_model()

    print(
        "Action space size:",
        action_encoder.size()
    )

    # ------------------------------------------------------
    # Create fresh RL optimizer
    # ------------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    # ------------------------------------------------------
    # Load RL1 model
    # ------------------------------------------------------

    checkpoint = load_previous_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=PREVIOUS_CHECKPOINT
    )

    print(
        "\nLoaded previous RL checkpoint."
    )

    for key in [
        "iteration",
        "completed_games",
        "incomplete_games",
        "white_wins",
        "black_wins",
        "draws",
        "new_samples"
    ]:

        if key in checkpoint:

            print(
                f"{key}:",
                checkpoint[key]
            )

    print(
        "RL learning rate:",
        LEARNING_RATE
    )

    # ------------------------------------------------------
    # Fresh replay buffer
    # ------------------------------------------------------

    replay_buffer = ReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY
    )

    print(
        "\nStarting with a FRESH replay buffer."
    )

    # ------------------------------------------------------
    # Self-play
    # ------------------------------------------------------

    self_play_stats = generate_self_play_data(
        model,
        replay_buffer
    )

    # ------------------------------------------------------
    # Check replay buffer
    # ------------------------------------------------------

    if len(replay_buffer) < TRAINING_BATCH_SIZE:

        raise RuntimeError(
            f"Only {len(replay_buffer)} training samples "
            f"were generated. Need at least "
            f"{TRAINING_BATCH_SIZE}."
        )

    # ------------------------------------------------------
    # Save replay buffer
    # ------------------------------------------------------

    save_replay_buffer(
        replay_buffer,
        REPLAY_BUFFER_CHECKPOINT
    )

    # ------------------------------------------------------
    # RL training
    # ------------------------------------------------------

    training_stats = train_model(
        model,
        optimizer,
        replay_buffer
    )

    # ------------------------------------------------------
    # Save RL checkpoint
    # ------------------------------------------------------

    save_rl_checkpoint(
        model=model,
        optimizer=optimizer,
        self_play_stats=self_play_stats,
        training_stats=training_stats
    )

    # ------------------------------------------------------
    # Finished
    # ------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        f"RL ITERATION {ITERATION} COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        "Model checkpoint:",
        OUTPUT_CHECKPOINT
    )

    print(
        "Replay buffer:",
        REPLAY_BUFFER_CHECKPOINT
    )


if __name__ == "__main__":
    main()
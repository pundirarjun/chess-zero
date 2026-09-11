import os
import sys
import numpy as np
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
from training.self_play import play_games
from training.replay_buffer import ReplayBuffer
from training.trainer import train_one_batch


# ==========================================================
# CONFIGURATION
# ==========================================================

# ==========================================================
# CHANGE ONLY THIS NUMBER FOR THE NEXT RL ITERATION.
#
# Example:
#   RL_ITERATION = 2  -> loads RL1, saves RL2
#   RL_ITERATION = 3  -> loads RL2, saves RL3
# ==========================================================

RL_ITERATION = 2

PREVIOUS_ITERATION = RL_ITERATION - 1

KAGGLE_CHECKPOINT_ROOT = (
    "/kaggle/input/datasets/arjunthakur9999/checkpoints"
)

LOCAL_CHECKPOINT_ROOT = (
    "checkpoints"
)

PREVIOUS_CHECKPOINT = os.path.join(
    KAGGLE_CHECKPOINT_ROOT,
    f"rl_iteration_{PREVIOUS_ITERATION}.pt"
)

PREVIOUS_REPLAY_BUFFER = os.path.join(
    KAGGLE_CHECKPOINT_ROOT,
    f"replay_buffer_rl{PREVIOUS_ITERATION}.pt"
)

OUTPUT_CHECKPOINT = os.path.join(
    LOCAL_CHECKPOINT_ROOT,
    f"rl_iteration_{RL_ITERATION}.pt"
)

OUTPUT_REPLAY_BUFFER = os.path.join(
    LOCAL_CHECKPOINT_ROOT,
    f"replay_buffer_rl{RL_ITERATION}.pt"
)


# ==========================================================
# SELF-PLAY CONFIGURATION
# ==========================================================

NUM_SELF_PLAY_GAMES = 10
NUM_SIMULATIONS = 50
MAX_MOVES = 300

# Global neural-network batch size used by multi-game MCTS.
MCTS_BATCH_SIZE = 128

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
TRAINING_STEPS = 50
LEARNING_RATE = 1e-4


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if device.type == "cuda":
    # Better convolution performance on NVIDIA GPUs.
    torch.backends.cudnn.benchmark = True

    # Safe TF32 acceleration for FP32 matrix/convolution operations.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # High-performance FP32 matmul selection.
    torch.set_float32_matmul_precision("high")


# ==========================================================
# CREATE MODEL
# ==========================================================

def create_model():

    action_encoder = ActionEncoder()

    model = ChessNet(
        action_space_size=action_encoder.size()
    )

    model.to(device)

    if device.type == "cuda":
        model.to(
            memory_format=torch.channels_last
        )

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
# CREATE AMP SCALER
# ==========================================================

def create_scaler():

    if device.type != "cuda":
        return None

    return torch.amp.GradScaler(
        "cuda"
    )


# ==========================================================
# LOAD PREVIOUS MODEL
# ==========================================================

def load_previous_checkpoint(
    model,
    optimizer,
    checkpoint_path
):

    if not os.path.exists(checkpoint_path):

        # Useful fallback when running outside Kaggle.
        local_path = os.path.join(
            LOCAL_CHECKPOINT_ROOT,
            os.path.basename(checkpoint_path)
        )

        if os.path.exists(local_path):
            checkpoint_path = local_path
        else:
            raise FileNotFoundError(
                f"Starting checkpoint not found:\n"
                f"{checkpoint_path}"
            )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # RL uses a fresh optimizer.
    for param_group in optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    model.to(device)

    if device.type == "cuda":
        model.to(
            memory_format=torch.channels_last
        )

    model.train()

    return checkpoint


# ==========================================================
# LOAD PREVIOUS REPLAY BUFFER
# ==========================================================

def load_previous_replay_buffer(
    path
):

    if not os.path.exists(path):

        local_path = os.path.join(
            LOCAL_CHECKPOINT_ROOT,
            os.path.basename(path)
        )

        if os.path.exists(local_path):
            path = local_path
        else:
            raise FileNotFoundError(
                f"Previous replay buffer not found:\n"
                f"{path}"
            )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(data, list):

        raise RuntimeError(
            "Invalid replay buffer format."
        )

    replay_buffer = ReplayBuffer(
        capacity=REPLAY_BUFFER_CAPACITY
    )

    replay_buffer.add(data)

    print(
        "\nPrevious replay buffer loaded."
    )

    print(
        "Previous samples:",
        len(replay_buffer)
    )

    return replay_buffer


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

    # IMPORTANT:
    # Generate all independent games together.
    # self_play.play_games() combines their MCTS neural-network
    # evaluations into shared GPU batches.
    results = play_games(
        model=model,
        num_games=NUM_SELF_PLAY_GAMES,
        num_simulations=NUM_SIMULATIONS,
        max_moves=MAX_MOVES,
        temperature=TEMPERATURE,
        temperature_moves=TEMPERATURE_MOVES,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        batch_size=MCTS_BATCH_SIZE
    )

    completed_games = 0
    incomplete_games = 0
    white_wins = 0
    black_wins = 0
    draws = 0
    new_samples = 0
    termination_counts = {}

    for game_number, result in enumerate(
        results,
        start=1
    ):

        termination_counts[
            result.termination
        ] = termination_counts.get(
            result.termination,
            0
        ) + 1

        if not result.completed:

            incomplete_games += 1

            print(
                f"Game {game_number}: "
                "Skipping incomplete game."
            )

            continue

        completed_games += 1

        if result.training_data:

            replay_buffer.add(
                result.training_data
            )

            new_samples += len(
                result.training_data
            )

        if result.result == 1:
            white_wins += 1

        elif result.result == -1:
            black_wins += 1

        elif result.result == 0:
            draws += 1

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

    scaler = create_scaler()

    for step in range(
        1,
        TRAINING_STEPS + 1
    ):

        batch_samples = replay_buffer.sample(
            TRAINING_BATCH_SIZE
        )

        # Build contiguous CPU arrays in one operation instead of
        # constructing many small tensors with torch.stack().
        states_np = np.asarray(
            [
                sample[0]
                for sample in batch_samples
            ],
            dtype=np.float32
        )

        policies_np = np.asarray(
            [
                sample[1]
                for sample in batch_samples
            ],
            dtype=np.float32
        )

        values_np = np.asarray(
            [
                sample[2]
                for sample in batch_samples
            ],
            dtype=np.float32
        )

        states = torch.from_numpy(
            states_np
        )

        policies = torch.from_numpy(
            policies_np
        )

        values = torch.from_numpy(
            values_np
        )

        if device.type == "cuda":
            states = states.pin_memory()
            policies = policies.pin_memory()
            values = values.pin_memory()

        states = states.to(
            device,
            non_blocking=(device.type == "cuda")
        )

        policies = policies.to(
            device,
            non_blocking=(device.type == "cuda")
        )

        values = values.to(
            device,
            non_blocking=(device.type == "cuda")
        )

        if device.type == "cuda":
            states = states.contiguous(
                memory_format=torch.channels_last
            )

        total_loss, policy_loss, value_loss = (
            train_one_batch(
                model=model,
                optimizer=optimizer,
                states=states,
                target_policy=policies,
                target_value=values,
                scaler=scaler
            )
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
            RL_ITERATION,

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
        RL_ITERATION
    )

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version:",
            torch.version.cuda
        )

        print(
            "Mixed precision: FP16"
        )

        print(
            "Channels last: enabled"
        )

        print(
            "TF32: enabled"
        )

    print(
        "Previous checkpoint:",
        PREVIOUS_CHECKPOINT
    )

    print(
        "Previous replay buffer:",
        PREVIOUS_REPLAY_BUFFER
    )

    print(
        "Output checkpoint:",
        OUTPUT_CHECKPOINT
    )

    print(
        "Output replay buffer:",
        OUTPUT_REPLAY_BUFFER
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
    # Create fresh optimizer
    # ------------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    # ------------------------------------------------------
    # Load previous model
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
    # Load previous replay buffer
    # ------------------------------------------------------

    replay_buffer = load_previous_replay_buffer(
        PREVIOUS_REPLAY_BUFFER
    )

    # ------------------------------------------------------
    # Generate new self-play data
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
            f"are available. Need at least "
            f"{TRAINING_BATCH_SIZE}."
        )

    # ------------------------------------------------------
    # Save updated replay buffer
    # ------------------------------------------------------

    save_replay_buffer(
        replay_buffer,
        OUTPUT_REPLAY_BUFFER
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
        f"RL ITERATION {RL_ITERATION} COMPLETE"
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
        OUTPUT_REPLAY_BUFFER
    )


if __name__ == "__main__":
    main()

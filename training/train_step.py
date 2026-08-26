import os

import torch

from training.self_play import play_game


# ==================================================
# DATASET
# ==================================================

class ChessDataset:

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        state, policy, value = self.samples[index]

        state = torch.from_numpy(state).float()
        policy = torch.from_numpy(policy).float()
        value = torch.tensor(value, dtype=torch.float32)

        return state, policy, value


# ==================================================
# ALPHAZERO LOSS
# ==================================================

def alpha_zero_loss(
    policy_logits,
    value_pred,
    target_policy,
    target_value
):

    # Policy loss
    policy_loss = -torch.sum(
        target_policy
        * torch.log_softmax(
            policy_logits,
            dim=1
        ),
        dim=1
    ).mean()

    # Value loss
    value_pred = value_pred.squeeze(-1)

    value_loss = torch.mean(
        (value_pred - target_value) ** 2
    )

    # Combined loss
    total_loss = (
        policy_loss
        + value_loss
    )

    return (
        total_loss,
        policy_loss,
        value_loss
    )


# ==================================================
# TRAIN FROM REPLAY BUFFER
# ==================================================

def train_from_replay_buffer(
    model,
    optimizer,
    replay_buffer,
    batch_size=32,
    training_steps=10
):

    if len(replay_buffer) < batch_size:

        raise ValueError(
            f"Replay buffer contains "
            f"{len(replay_buffer)} samples, "
            f"but batch size is {batch_size}."
        )

    total_losses = []
    policy_losses = []
    value_losses = []

    for step in range(training_steps):

        samples = replay_buffer.sample(
            batch_size
        )

        dataset = ChessDataset(
            samples
        )

        states = []
        policies = []
        values = []

        for i in range(len(dataset)):

            state, policy, value = dataset[i]

            states.append(state)
            policies.append(policy)
            values.append(value)

        states = torch.stack(
            states
        )

        policies = torch.stack(
            policies
        )

        values = torch.stack(
            values
        )

        # --------------------------------------------------
        # Move data to model device
        # --------------------------------------------------

        device = next(
            model.parameters()
        ).device

        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        model.train()

        optimizer.zero_grad()

        predicted_policy, predicted_value = model(
            states
        )

        total_loss, policy_loss, value_loss = (
            alpha_zero_loss(
                predicted_policy,
                predicted_value,
                policies,
                values
            )
        )

        total_loss.backward()

        optimizer.step()

        total_losses.append(
            total_loss.item()
        )

        policy_losses.append(
            policy_loss.item()
        )

        value_losses.append(
            value_loss.item()
        )

    return {
        "total_loss": total_losses,
        "policy_loss": policy_losses,
        "value_loss": value_losses
    }


# ==================================================
# CHECKPOINT
# ==================================================

def save_checkpoint(
    model,
    optimizer,
    iteration,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(
        checkpoint,
        path
    )


# ==================================================
# FULL TRAINING ITERATION
# ==================================================

def run_training_iteration(
    model,
    optimizer,
    replay_buffer,
    iteration,
    num_games=3,
    num_simulations=100,
    max_moves=100,
    batch_size=32,
    training_steps=10
):

    print(
        f"\n========== Iteration {iteration} =========="
    )

    # ==================================================
    # SELF-PLAY
    # ==================================================

    print(
        "\nGenerating self-play games..."
    )

    successful_games = 0

    for game_number in range(
        num_games
    ):

        print(
            f"\nGame {game_number + 1}/{num_games}"
        )

        result = play_game(
            model=model,
            num_simulations=num_simulations,
            max_moves=max_moves
        )

        print(
            "Samples:",
            len(result.training_data)
        )

        print(
            "Game result:",
            result.result,
            "| Termination:",
            result.termination,
            "| Moves:",
            result.moves_played
        )

        # --------------------------------------------------
        # Only completed games enter replay buffer
        # --------------------------------------------------

        if result.completed:

            replay_buffer.add(
                result.training_data
            )

            successful_games += 1

        else:

            print(
                "Game was incomplete. "
                "Samples discarded."
            )

    # ==================================================
    # REPLAY BUFFER
    # ==================================================

    print(
        "\nReplay buffer size:",
        len(replay_buffer)
    )

    print(
        "Completed games:",
        successful_games,
        "/",
        num_games
    )

    # --------------------------------------------------
    # Check enough data
    # --------------------------------------------------

    if len(replay_buffer) < batch_size:

        raise ValueError(
            f"Replay buffer contains "
            f"{len(replay_buffer)} samples, "
            f"but batch size is {batch_size}."
        )

    # ==================================================
    # TRAINING
    # ==================================================

    print(
        "\nTraining..."
    )

    losses = train_from_replay_buffer(
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        training_steps=training_steps
    )

    # ==================================================
    # PRINT LOSSES
    # ==================================================

    for step in range(
        training_steps
    ):

        print(
            f"Step {step + 1}: "
            f"Total={losses['total_loss'][step]:.6f} | "
            f"Policy={losses['policy_loss'][step]:.6f} | "
            f"Value={losses['value_loss'][step]:.6f}"
        )

    # ==================================================
    # SAVE CHECKPOINT
    # ==================================================

    checkpoint_path = (
        f"checkpoints/"
        f"iteration_{iteration}.pt"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        path=checkpoint_path
    )

    print(
        "\nCheckpoint saved:",
        checkpoint_path
    )

    return losses
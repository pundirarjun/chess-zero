from training.self_play import play_game
from training.replay_buffer import ReplayBuffer
from training.train_step import train_from_replay_buffer
from training.checkpoint import save_checkpoint


def run_training_iteration(
    model,
    optimizer,
    replay_buffer,
    iteration,
    num_games=3,
    num_simulations=10,
    max_moves=30,
    batch_size=8,
    training_steps=10
):

    print(
        f"\n========== Iteration {iteration} =========="
    )

    # --------------------------------------------------
    # Self-play
    # --------------------------------------------------

    print("\nGenerating self-play games...")

    for game_number in range(num_games):

        print(
            f"Game {game_number + 1}/{num_games}"
        )

        samples = play_game(
            model=model,
            num_simulations=num_simulations,
            max_moves=max_moves
        )

        replay_buffer.add(
            samples
        )

        print(
            "Samples:",
            len(samples)
        )

    print(
        "\nReplay buffer size:",
        len(replay_buffer)
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print("\nTraining...")

    losses = train_from_replay_buffer(
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        training_steps=training_steps
    )

    # --------------------------------------------------
    # Print training results
    # --------------------------------------------------

    for step in range(training_steps):

        print(
            f"Step {step + 1}: "
            f"Total={losses['total_loss'][step]:.6f} | "
            f"Policy={losses['policy_loss'][step]:.6f} | "
            f"Value={losses['value_loss'][step]:.6f}"
        )

    # --------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------

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
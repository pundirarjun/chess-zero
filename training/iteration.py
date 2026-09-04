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
    mcts_batch_size=64,
    training_batch_size=8,
    training_steps=10,
    temperature=1.0,
    temperature_moves=20,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25
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

    completed_games = 0
    truncated_games = 0

    total_samples_added = 0

    # --------------------------------------------------
    # Generate games
    # --------------------------------------------------

    for game_number in range(
        num_games
    ):

        print(
            f"\nGame {game_number + 1}/{num_games}"
        )

        game_result = play_game(

            model=model,

            num_simulations=num_simulations,

            max_moves=max_moves,

            temperature=temperature,

            temperature_moves=temperature_moves,

            dirichlet_alpha=dirichlet_alpha,

            dirichlet_epsilon=dirichlet_epsilon,

            batch_size=mcts_batch_size
        )

        # --------------------------------------------------
        # Training samples
        # --------------------------------------------------

        samples = game_result.training_data

        if game_result.completed:

            completed_games += 1

        else:

            truncated_games += 1

        # --------------------------------------------------
        # Add only valid completed-game data
        # --------------------------------------------------

        if samples:

            replay_buffer.add(
                samples
            )

            total_samples_added += len(
                samples
            )

        # --------------------------------------------------
        # Game statistics
        # --------------------------------------------------

        print(
            "Samples:",
            len(samples)
        )

        print(
            "Game result:",
            game_result.result,
            "| Termination:",
            game_result.termination,
            "| Moves:",
            game_result.moves_played,
            "| Completed:",
            game_result.completed
        )

    # ==================================================
    # SELF-PLAY SUMMARY
    # ==================================================

    print(
        "\n========== SELF-PLAY SUMMARY =========="
    )

    print(
        "Games generated:",
        num_games
    )

    print(
        "Completed games:",
        completed_games
    )

    print(
        "Truncated games:",
        truncated_games
    )

    print(
        "Samples added:",
        total_samples_added
    )

    print(
        "Replay buffer size:",
        len(replay_buffer)
    )

    # ==================================================
    # SAFETY CHECK
    # ==================================================

    if len(replay_buffer) == 0:

        print(
            "\nNo training data available."
        )

        print(
            "Skipping training and checkpoint."
        )

        return None

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

        batch_size=training_batch_size,

        training_steps=training_steps

    )

    # ==================================================
    # PRINT TRAINING RESULTS
    # ==================================================

    print(
        "\n========== TRAINING RESULTS =========="
    )

    # --------------------------------------------------
    # Safely determine number of recorded steps
    # --------------------------------------------------

    recorded_steps = len(
        losses["total_loss"]
    )

    for step in range(
        recorded_steps
    ):

        print(

            f"Step {step + 1}: "

            f"Total={losses['total_loss'][step]:.6f} | "

            f"Policy={losses['policy_loss'][step]:.6f} | "

            f"Value={losses['value_loss'][step]:.6f}"

        )

    # ==================================================
    # CHECKPOINT
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

    # ==================================================
    # ITERATION SUMMARY
    # ==================================================

    print(
        "\n========== ITERATION SUMMARY =========="
    )

    print(
        "Iteration:",
        iteration
    )

    print(
        "Completed games:",
        completed_games,
        "/",
        num_games
    )

    print(
        "Truncated games:",
        truncated_games
    )

    print(
        "New samples:",
        total_samples_added
    )

    print(
        "Replay buffer:",
        len(replay_buffer)
    )

    print(
        "MCTS simulations/game:",
        num_simulations
    )

    print(
        "MCTS batch size:",
        mcts_batch_size
    )

    print(
        "Training batch size:",
        training_batch_size
    )

    print(
        "Training steps:",
        training_steps
    )

    return losses
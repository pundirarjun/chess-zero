from training.self_play import play_games

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

    # --------------------------------------------------
    # Generate all games together.
    #
    # This uses multi-game batched MCTS so neural-network
    # evaluations from different games can share GPU
    # batches.
    # --------------------------------------------------

    results = play_games(

        model=model,

        num_games=num_games,

        num_simulations=num_simulations,

        max_moves=max_moves,

        temperature=temperature,

        temperature_moves=temperature_moves,

        dirichlet_alpha=dirichlet_alpha,

        dirichlet_epsilon=dirichlet_epsilon,

        batch_size=mcts_batch_size

    )

    # ==================================================
    # PROCESS SELF-PLAY RESULTS
    # ==================================================

    completed_games = 0
    truncated_games = 0

    total_samples_added = 0

    for game_number, game_result in enumerate(
        results
    ):

        samples = game_result.training_data

        # --------------------------------------------------
        # Game statistics
        # --------------------------------------------------

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

        print(
            f"\nGame {game_number + 1}/{num_games}"
        )

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
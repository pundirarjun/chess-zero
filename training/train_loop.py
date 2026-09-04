from training.iteration import run_training_iteration


def run_training_loop(
    model,
    optimizer,
    replay_buffer,
    start_iteration=1,
    num_iterations=3,
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

    history = []

    for iteration in range(
        start_iteration,
        start_iteration + num_iterations
    ):

        losses = run_training_iteration(
            model=model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            iteration=iteration,
            num_games=num_games,
            num_simulations=num_simulations,
            max_moves=max_moves,
            mcts_batch_size=mcts_batch_size,
            training_batch_size=training_batch_size,
            training_steps=training_steps,
            temperature=temperature,
            temperature_moves=temperature_moves,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )

        # No training occurred
        if losses is None:
            history.append(
                {
                    "iteration": iteration,
                    "replay_buffer_size": len(replay_buffer),
                    "average_total_loss": None,
                    "average_policy_loss": None,
                    "average_value_loss": None
                }
            )

            print(
                "\nIteration skipped: "
                "no training data available."
            )

            continue

        average_total_loss = (
            sum(losses["total_loss"])
            / len(losses["total_loss"])
        )

        average_policy_loss = (
            sum(losses["policy_loss"])
            / len(losses["policy_loss"])
        )

        average_value_loss = (
            sum(losses["value_loss"])
            / len(losses["value_loss"])
        )

        history.append(
            {
                "iteration": iteration,
                "replay_buffer_size": len(replay_buffer),
                "average_total_loss":
                    average_total_loss,
                "average_policy_loss":
                    average_policy_loss,
                "average_value_loss":
                    average_value_loss
            }
        )

        print("\nIteration summary:")

        print(
            "Iteration:",
            iteration
        )

        print(
            "Replay buffer:",
            len(replay_buffer)
        )

        print(
            "Average total loss:",
            average_total_loss
        )

        print(
            "Average policy loss:",
            average_policy_loss
        )

        print(
            "Average value loss:",
            average_value_loss
        )

    return history
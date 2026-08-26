import chess

from environment.action_encoder import ActionEncoder
from mcts.node import Node
from mcts.mcts import MCTS


def select_move(
    model,
    board,
    num_simulations,
    action_encoder
):

    root = Node(board)

    mcts = MCTS(
        model=model,
        action_encoder=action_encoder
    )

    root.expand(
        model,
        action_encoder
    )

    for _ in range(
        max(0, num_simulations - 1)
    ):

        mcts.run_simulation(
            root
        )

    return mcts.select_action_with_temperature(
        root,
        temperature=0
    )


def play_match(
    white_model,
    black_model,
    num_simulations=10,
    max_moves=100
):

    board = chess.Board()

    action_encoder = ActionEncoder()

    move_number = 0

    move_history = []

    while not board.is_game_over():

        if move_number >= max_moves:

            return {
                "result": 0,
                "termination": "max_moves",
                "moves": move_history
            }

        if board.turn == chess.WHITE:

            model = white_model

        else:

            model = black_model

        move = select_move(
            model=model,
            board=board,
            num_simulations=num_simulations,
            action_encoder=action_encoder
        )

        move_history.append(
            move.uci()
        )

        board.push(move)

        move_number += 1

    outcome = board.outcome(
    claim_draw=True
    )

    if outcome is None:

        return {
            "result": 0,
            "termination": "unknown",
            "moves": move_history
        }

    if outcome.winner is None:

        result = 0

    elif outcome.winner == chess.WHITE:

        result = 1

    else:

        result = -1

    return {
        "result": result,
        "termination": str(
            outcome.termination
        ),
        "moves": move_history
    }


def evaluate_models(
    old_model,
    new_model,
    num_games=10,
    num_simulations=10,
    max_moves=100
):

    new_wins = 0
    old_wins = 0
    draws = 0

    terminations = {}

    for game in range(num_games):

        print(
            f"Evaluation game "
            f"{game + 1}/{num_games}"
        )

        if game % 2 == 0:

            result = play_match(
                white_model=new_model,
                black_model=old_model,
                num_simulations=num_simulations,
                max_moves=max_moves
            )

            if result["result"] == 1:

                new_wins += 1

            elif result["result"] == -1:

                old_wins += 1

            else:

                draws += 1

        else:

            result = play_match(
                white_model=old_model,
                black_model=new_model,
                num_simulations=num_simulations,
                max_moves=max_moves
            )

            if result["result"] == 1:

                old_wins += 1

            elif result["result"] == -1:

                new_wins += 1

            else:

                draws += 1

        termination = result["termination"]

        print(
            "Moves:",
            " ".join(result["moves"])
        )

        terminations[termination] = (
            terminations.get(
                termination,
                0
            ) + 1
        )

        print(
            "Result:",
            result["result"],
            "| Termination:",
            termination
        )

    return {
        "new_wins": new_wins,
        "old_wins": old_wins,
        "draws": draws,
        "terminations": terminations
    }
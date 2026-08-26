from dataclasses import dataclass

import numpy as np
import chess

from environment.state_encoder import StateEncoder
from environment.action_encoder import ActionEncoder

from mcts.node import Node
from mcts.mcts import MCTS


@dataclass
class TrainingSample:

    state: np.ndarray
    policy: np.ndarray
    player: int


@dataclass
class SelfPlayResult:

    training_data: list
    result: int | None
    termination: str
    moves_played: int
    completed: bool


class SelfPlayGame:

    def __init__(self):

        self.samples = []

    def add_position(
        self,
        board,
        policy
    ):

        state = StateEncoder.encode(board)

        player = (
            1
            if board.turn == chess.WHITE
            else -1
        )

        sample = TrainingSample(

            state=state,

            policy=np.array(
                policy,
                dtype=np.float32
            ),

            player=player
        )

        self.samples.append(sample)

    def get_training_data(
        self,
        result
    ):

        training_data = []

        for sample in self.samples:

            if result == 0:

                value = 0.0

            elif result == 1:

                value = float(
                    sample.player
                )

            else:

                value = float(
                    -sample.player
                )

            training_data.append(

                (
                    sample.state,
                    sample.policy,
                    value
                )
            )

        return training_data


def play_game(

    model,

    num_simulations=100,

    max_moves=200,

    temperature=1.0,

    temperature_moves=20,

    dirichlet_alpha=0.3,

    dirichlet_epsilon=0.25

):

    board = chess.Board()

    action_encoder = ActionEncoder()

    mcts = MCTS(

        model=model,

        action_encoder=action_encoder
    )

    game = SelfPlayGame()

    move_number = 1

    result = None

    completed = False

    termination = "UNKNOWN"


    # ==================================================
    # SELF-PLAY LOOP
    # ==================================================

    while not board.is_game_over(
        claim_draw=True
    ):

        # Safety limit.
        #
        # Use > rather than >= so max_moves
        # means the actual maximum number of moves.

        if move_number > max_moves:

            result = None

            termination = "MAX_MOVES"

            break


        # ==================================================
        # CREATE MCTS ROOT
        # ==================================================

        root = Node(board)


        # ==================================================
        # EXPAND ROOT
        # ==================================================

        root.expand(

            model,

            action_encoder
        )


        # ==================================================
        # DIRICHLET EXPLORATION
        # ==================================================

        mcts.add_dirichlet_noise(

            root,

            alpha=dirichlet_alpha,

            epsilon=dirichlet_epsilon
        )


        # ==================================================
        # MCTS SIMULATIONS
        # ==================================================

        for _ in range(

            max(
                0,
                num_simulations - 1
            )

        ):

            mcts.run_simulation(

                root
            )


        # ==================================================
        # MCTS POLICY TARGET
        # ==================================================

        policy = mcts.get_policy_target(

            root
        )


        # ==================================================
        # STORE POSITION
        # ==================================================

        game.add_position(

            board,

            policy
        )


        # ==================================================
        # TEMPERATURE
        # ==================================================

        if move_number <= temperature_moves:

            current_temperature = temperature

        else:

            current_temperature = 0.0


        # ==================================================
        # SELECT MOVE
        # ==================================================

        move = mcts.select_action_with_temperature(

            root,

            temperature=current_temperature
        )


        # ==================================================
        # PLAY MOVE
        # ==================================================

        board.push(move)

        move_number += 1


    # ==================================================
    # GAME ACTUALLY TERMINATED
    # ==================================================

    if board.is_game_over(
        claim_draw=True
    ):

        completed = True

        outcome = board.outcome(
            claim_draw=True
        )

        if outcome is None:

            result = 0

            termination = "UNKNOWN"

        else:

            termination = str(
                outcome.termination
            )

            if outcome.winner is None:

                result = 0

            elif outcome.winner == chess.WHITE:

                result = 1

            else:

                result = -1


    # ==================================================
    # GAME WAS TRUNCATED
    # ==================================================

    else:

        completed = False

        result = None

        termination = "MAX_MOVES"


    # ==================================================
    # STATISTICS
    # ==================================================

    moves_played = len(
        game.samples
    )

    print(
        "\nSelf-play termination:"
    )

    print(
        "Moves played:",
        moves_played
    )

    print(
        "Result:",
        result
    )

    print(
        "Termination:",
        termination
    )

    print(
        "Completed:",
        completed
    )

    print(
        "Board outcome:",
        board.outcome(
            claim_draw=True
        )
    )


    # ==================================================
    # DISCARD INCOMPLETE GAME
    # ==================================================

    if not completed:

        print(
            "Game truncated before terminal result."
        )

        return SelfPlayResult(

            training_data=[],

            result=None,

            termination=termination,

            moves_played=moves_played,

            completed=False
        )


    # ==================================================
    # CREATE TRAINING DATA
    # ==================================================

    training_data = game.get_training_data(
        result
    )


    return SelfPlayResult(

        training_data=training_data,

        result=result,

        termination=termination,

        moves_played=moves_played,

        completed=True
    )
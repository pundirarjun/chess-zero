from dataclasses import dataclass

import numpy as np
import chess

from environment.state_encoder import StateEncoder
from environment.action_encoder import ActionEncoder

from mcts.node import Node
from mcts.mcts import MCTS


# ==========================================================
# TRAINING SAMPLE
# ==========================================================

@dataclass
class TrainingSample:

    state: np.ndarray
    policy: np.ndarray
    player: int


# ==========================================================
# SELF-PLAY RESULT
# ==========================================================

@dataclass
class SelfPlayResult:

    training_data: list
    result: int | None
    termination: str
    moves_played: int
    completed: bool


# ==========================================================
# SELF-PLAY GAME
# ==========================================================

class SelfPlayGame:

    def __init__(self):

        self.samples = []

    # ======================================================
    # STORE POSITION
    # ======================================================

    def add_position(
        self,
        board,
        policy
    ):

        # Encode board state.

        state = StateEncoder.encode(
            board
        )

        # Store which player was to move.

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

        self.samples.append(
            sample
        )

    # ======================================================
    # CREATE TRAINING DATA
    # ======================================================

    def get_training_data(
        self,
        result
    ):

        training_data = []

        for sample in self.samples:

            # --------------------------------------------------
            # Draw
            # --------------------------------------------------

            if result == 0:

                value = 0.0

            # --------------------------------------------------
            # White won
            # --------------------------------------------------

            elif result == 1:

                value = float(
                    sample.player
                )

            # --------------------------------------------------
            # Black won
            # --------------------------------------------------

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


# ==========================================================
# PLAY ONE SELF-PLAY GAME
# ==========================================================

def play_game(

    model,

    num_simulations=100,

    max_moves=200,

    temperature=1.0,

    temperature_moves=20,

    dirichlet_alpha=0.3,

    dirichlet_epsilon=0.25,

    batch_size=64

):

    # ======================================================
    # INITIALIZATION
    # ======================================================

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

    # ======================================================
    # SELF-PLAY LOOP
    # ======================================================

    while not board.is_game_over(
        claim_draw=True
    ):

        # ==================================================
        # SAFETY LIMIT
        # ==================================================

        if move_number > max_moves:

            result = None

            termination = "MAX_MOVES"

            break

        # ==================================================
        # CREATE ROOT
        # ==================================================

        root = Node(
            board
        )

        # ==================================================
        # ROOT EXPANSION
        #
        # Root expansion is initialization.
        # It is NOT counted as an MCTS simulation.
        # ==================================================

        root.expand(

            model,

            action_encoder

        )

        # ==================================================
        # DIRICHLET EXPLORATION
        #
        # Used during self-play.
        # ==================================================

        mcts.add_dirichlet_noise(

            root,

            alpha=dirichlet_alpha,

            epsilon=dirichlet_epsilon

        )

        # ==================================================
        # BATCHED MCTS SEARCH
        #
        # num_simulations means ACTUAL simulations.
        #
        # Example:
        #
        # num_simulations=500
        #
        # means:
        #
        # root initialization
        # +
        # 500 actual simulations
        # ==================================================

        if num_simulations > 0:

            mcts.search_batched(

                root,

                num_simulations=num_simulations,

                batch_size=batch_size

            )

        # ==================================================
        # GET MCTS POLICY TARGET
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

        move = (
            mcts.select_action_with_temperature(

                root,

                temperature=current_temperature

            )
        )

        # ==================================================
        # PLAY MOVE
        # ==================================================

        print(
            f"{move_number}: {move}"
        )

        board.push(
            move
        )

        move_number += 1

    # ======================================================
    # CHECK WHETHER GAME ACTUALLY TERMINATED
    # ======================================================

    if board.is_game_over(
        claim_draw=True
    ):

        completed = True

        outcome = board.outcome(
            claim_draw=True
        )

        # --------------------------------------------------
        # Safety fallback
        # --------------------------------------------------

        if outcome is None:

            result = 0

            termination = "UNKNOWN"

        else:

            termination = str(
                outcome.termination
            )

            # --------------------------------------------------
            # Draw
            # --------------------------------------------------

            if outcome.winner is None:

                result = 0

            # --------------------------------------------------
            # White won
            # --------------------------------------------------

            elif outcome.winner == chess.WHITE:

                result = 1

            # --------------------------------------------------
            # Black won
            # --------------------------------------------------

            else:

                result = -1

    # ======================================================
    # GAME WAS TRUNCATED
    # ======================================================

    else:

        completed = False

        result = None

        termination = "MAX_MOVES"

    # ======================================================
    # STATISTICS
    # ======================================================

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

    # ======================================================
    # DISCARD INCOMPLETE GAME
    #
    # IMPORTANT:
    #
    # MAX_MOVES does NOT automatically mean draw.
    #
    # We don't know the actual game result, so we discard
    # the game from the RL training data.
    # ======================================================

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

    # ======================================================
    # CREATE TRAINING DATA
    # ======================================================

    training_data = (
        game.get_training_data(
            result
        )
    )

    # ======================================================
    # RETURN RESULT
    # ======================================================

    return SelfPlayResult(

        training_data=training_data,

        result=result,

        termination=termination,

        moves_played=moves_played,

        completed=True

    )
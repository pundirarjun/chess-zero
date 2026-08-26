import torch

from environment.state_encoder import StateEncoder


def get_policy_for_board(
    board,
    model,
    action_encoder
):

    if board.is_game_over(
        claim_draw=True
    ):

        return {}


    model.eval()


    state = StateEncoder.encode(
        board
    )


    device = next(model.parameters()).device

    state_tensor = torch.from_numpy(
        state
    ).unsqueeze(0).to(device)


    with torch.no_grad():

        policy_logits, _ = model(
            state_tensor
        )


    policy_logits = policy_logits[0]


    legal_moves = list(
        board.legal_moves
    )


    if not legal_moves:

        return {}


    legal_action_ids = [

        action_encoder.encode(move)

        for move in legal_moves

    ]


    legal_logits = policy_logits[
        legal_action_ids
    ]


    legal_priors = torch.softmax(
        legal_logits,
        dim=0
    )


    # Numerical safety
    legal_priors = legal_priors / (
        legal_priors.sum()
    )


    return {

        move: prior.item()

        for move, prior in zip(
            legal_moves,
            legal_priors
        )

    }


def get_value_for_board(
    board,
    model
):

    model.eval()

    state = StateEncoder.encode(board)

    device = next(model.parameters()).device

    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

    with torch.no_grad():
        _, value = model(state_tensor)

    return value.item()


def get_policy_and_value_for_board(
    board,
    model,
    action_encoder
):

    if board.is_game_over(claim_draw=True):
        return {}, None

    model.eval()

    state = StateEncoder.encode(board)

    device = next(model.parameters()).device

    state_tensor = torch.from_numpy(
        state
    ).unsqueeze(0).float().to(device)

    with torch.no_grad():

        policy_logits, value = model(
            state_tensor
        )

    policy_logits = policy_logits[0]

    legal_moves = list(
        board.legal_moves
    )

    if not legal_moves:
        return {}, value.item()

    legal_action_ids = [
        action_encoder.encode(move)
        for move in legal_moves
    ]

    legal_logits = policy_logits[
        legal_action_ids
    ]

    legal_priors = torch.softmax(
        legal_logits,
        dim=0
    )

    legal_priors = legal_priors / legal_priors.sum()

    policy = {
        move: prior.item()
        for move, prior in zip(
            legal_moves,
            legal_priors
        )
    }

    return policy, value.item()
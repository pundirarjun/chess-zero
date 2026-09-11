import torch

from environment.gpu_chess import GPUChess, _action_tables, _signed_u64


def action_id(from_sq, to_sq, promotion=0):
    t = _action_tables()
    hits = [i for i, (f, to, p) in enumerate(zip(t["from_sq"], t["to_sq"], t["promo"])) if int(f) == from_sq and int(to) == to_sq and int(p) == promotion]
    assert len(hits) == 1
    return hits[0]


def test_initial_position():
    g = GPUChess("cpu", 1)
    legal = g.legal_move_mask()[0]
    assert int(legal.sum()) == 20
    assert not bool(g.is_in_check()[0])


def test_e2_e4():
    g = GPUChess("cpu", 1)
    g = g.push_actions(torch.tensor([action_id(12, 28)]))
    assert bool(g.turn[0])
    assert int(g.ep_square[0]) == 20
    assert int(g.legal_move_mask()[0].sum()) == 20


def test_pinned_piece():
    g = GPUChess("cpu", 1)
    g.pieces.zero_()
    g.pieces[0, 5] = _signed_u64(1 << 4)   # white king e1
    g.pieces[0, 3] = _signed_u64(1 << 12)  # white rook e2
    g.pieces[0, 9] = _signed_u64(1 << 60)  # black rook e8
    g.pieces[0, 11] = _signed_u64(1 << 56) # black king a8
    g.castling.zero_()
    legal = g.legal_move_mask()[0]
    t = _action_tables()
    for a in legal.nonzero().flatten().tolist():
        if int(t["from_sq"][a]) == 12:
            assert int(t["to_sq"][a]) % 8 == 4


def test_illegal_en_passant_exposes_king():
    g = GPUChess("cpu", 1)
    g.pieces.zero_()
    g.pieces[0, 5] = _signed_u64(1 << 4)    # white king e1
    g.pieces[0, 0] = _signed_u64(1 << 36)   # white pawn e5
    g.pieces[0, 6] = _signed_u64(1 << 35)   # black pawn d5
    g.pieces[0, 9] = _signed_u64(1 << 60)   # black rook e8
    g.pieces[0, 11] = _signed_u64(1 << 56)
    g.castling.zero_()
    g.ep_square[0] = 43  # d6
    legal = g.legal_move_mask()[0]
    ep = action_id(36, 43)
    assert not bool(legal[ep])


def test_promotion():
    g = GPUChess("cpu", 1)
    g.pieces.zero_()
    g.pieces[0, 5] = _signed_u64(1 << 4)
    g.pieces[0, 0] = _signed_u64(1 << 48)
    g.pieces[0, 11] = _signed_u64(1 << 60)
    g.castling.zero_()
    legal = g.legal_move_mask()[0]
    assert all(bool(legal[action_id(48, 56, p)]) for p in (2, 3, 4, 5))


def test_model_state_shape():
    g = GPUChess("cpu", 2)
    x = g.to_model_input()
    assert x.shape == (2, 18, 8, 8)
    assert x.dtype == torch.float32

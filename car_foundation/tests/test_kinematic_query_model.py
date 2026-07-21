import pytest
import torch

from car_foundation.models import (
    TorchTransformerDecoderCurrentStateMLP,
    TorchTransformerDecoderKinematicQueryMLP,
)


def _make_base_model():
    return TorchTransformerDecoderCurrentStateMLP(
        state_dim=5,
        action_dim=2,
        output_dim=4,
        latent_dim=16,
        num_heads=4,
        num_layers=1,
        device=torch.device("cpu"),
        dropout=0.0,
        history_length=250,
        prediction_length=50,
        compressed_history_length=42,
        current_dim=4,
        fusion_hidden_dim=8,
    )


def _make_query_model():
    return TorchTransformerDecoderKinematicQueryMLP(
        state_dim=5,
        action_dim=2,
        output_dim=4,
        latent_dim=16,
        num_heads=4,
        num_layers=1,
        device=torch.device("cpu"),
        dropout=0.0,
        history_length=250,
        prediction_length=50,
        compressed_history_length=42,
        current_dim=4,
        fusion_hidden_dim=8,
        nominal_state_dim=5,
        nominal_transition_dim=4,
        query_hidden_dim=8,
    )


def _inputs():
    torch.manual_seed(11)
    history = torch.randn(3, 250, 7)
    action = torch.randn(3, 50, 2)
    context = torch.randn(3, 4)
    nominal_state = torch.randn(3, 50, 5)
    nominal_transition = torch.randn(3, 50, 4)
    return history, action, context, nominal_state, nominal_transition


def test_query_model_loads_base_checkpoint_with_only_new_branch_missing():
    torch.manual_seed(3)
    base = _make_base_model()
    query = _make_query_model()
    incompatible = query.load_state_dict(base.state_dict(), strict=False)

    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "kinematic_query_fusion.0.weight",
        "kinematic_query_fusion.0.bias",
        "kinematic_query_fusion.2.weight",
        "kinematic_query_fusion.2.bias",
    }


def test_zero_initialized_query_preserves_base_embedding_exactly():
    torch.manual_seed(5)
    base = _make_base_model()
    query = _make_query_model()
    query.load_state_dict(base.state_dict(), strict=False)
    history, action, context, nominal_state, nominal_transition = _inputs()

    base_embedding = base._build_action_emb(history, action, context)
    query_embedding = query._build_action_emb(
        history,
        action,
        context,
        nominal_state,
        nominal_transition,
    )
    torch.testing.assert_close(query_embedding, base_embedding, rtol=0, atol=0)


def test_query_branch_becomes_nonzero_after_optimization_step():
    torch.manual_seed(7)
    query = _make_query_model()
    history, action, context, nominal_state, nominal_transition = _inputs()
    before = query._build_action_emb(
        history,
        action,
        context,
        nominal_state,
        nominal_transition,
    ).detach()
    optimizer = torch.optim.SGD(query.kinematic_query_fusion.parameters(), lr=0.1)
    output = query._build_action_emb(
        history,
        action,
        context,
        nominal_state,
        nominal_transition,
    )
    loss = output.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = query._build_action_emb(
        history,
        action,
        context,
        nominal_state,
        nominal_transition,
    ).detach()

    assert not torch.equal(after, before)


def test_query_requires_state_and_transition_together():
    query = _make_query_model()
    history, action, context, nominal_state, _ = _inputs()
    with pytest.raises(ValueError, match="must be provided together"):
        query._build_action_emb(
            history,
            action,
            context,
            nominal_state=nominal_state,
            nominal_transition=None,
        )

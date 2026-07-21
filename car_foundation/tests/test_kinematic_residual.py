from types import SimpleNamespace

import pytest
import torch

from car_foundation.kinematic_residual import (
    KinematicBicycleParams,
    NuPlanKinematicResidualDataset,
    apply_consistent_transition,
    apply_observable_delta,
    kinematic_consistent_transition,
    kinematic_delta,
    multi_horizon_rollout_loss,
    rollout_metric_squared_errors,
    rollout_consistent_transition_sequence,
    rollout_kinematic_residual_consistent,
    rollout_nominal_kinematic_consistent,
    states_relative_to_initial_body,
)


def test_straight_kinematic_step():
    params = KinematicBicycleParams(
        dt=0.05, wheelbase=3.9, steering_ratio=25.0, steering_offset=0.0
    )
    state = torch.tensor([[1.0, 2.0, 0.0, 10.0, 0.0]])
    action = torch.tensor([[0.0, 0.0]])
    delta = kinematic_delta(state, action, params)
    torch.testing.assert_close(delta, torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(
        apply_observable_delta(state, delta),
        torch.tensor([[1.5, 2.0, 0.0, 10.0, 0.0]]),
    )


def test_body_delta_is_rotated_to_world():
    state = torch.tensor([[0.0, 0.0, torch.pi / 2, 4.0, 0.0]])
    delta = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    result = apply_observable_delta(state, delta)
    torch.testing.assert_close(result[:, :2], torch.tensor([[0.0, 1.0]]), atol=1e-6, rtol=0)


def test_consistent_transition_integrates_current_yawrate():
    params = KinematicBicycleParams(dt=0.05)
    state = torch.tensor([[0.0, 0.0, 0.2, 10.0, 0.4]])
    transition = torch.tensor([[0.5, 0.0, 0.1, 0.3]])
    result = apply_consistent_transition(state, transition, params)
    torch.testing.assert_close(result[:, 2], torch.tensor([0.22]))
    torch.testing.assert_close(result[:, 3:], torch.tensor([[10.1, 0.7]]))


def test_consistent_rollout_uses_predicted_yawrate_on_following_step():
    params = KinematicBicycleParams(dt=0.05)
    state = torch.tensor([[0.0, 0.0, 0.0, 10.0, 0.2]])
    transitions = torch.tensor(
        [[[0.5, 0.0, 0.0, 0.2], [0.5, 0.0, 0.0, 0.0]]]
    )
    result = rollout_consistent_transition_sequence(state, transitions, params)
    torch.testing.assert_close(result[:, :, 2], torch.tensor([[0.01, 0.03]]))


def test_nominal_rollout_returns_relative_states_and_transitions():
    params = KinematicBicycleParams(
        dt=0.05, wheelbase=3.9, steering_ratio=25.0, steering_offset=0.0
    )
    initial = torch.tensor([10.0, -3.0, torch.pi / 2, 4.0, 0.0])
    action = torch.zeros(3, 2)
    states, transitions = rollout_nominal_kinematic_consistent(
        initial, action, params
    )
    relative = states_relative_to_initial_body(initial, states)

    assert states.shape == (3, 5)
    assert transitions.shape == (3, 4)
    torch.testing.assert_close(
        transitions,
        torch.tensor(
            [[0.2, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]]
        ),
    )
    torch.testing.assert_close(
        relative,
        torch.tensor(
            [[0.2, 0.0, 0.0, 4.0, 0.0], [0.4, 0.0, 0.0, 4.0, 0.0], [0.6, 0.0, 0.0, 4.0, 0.0]]
        ),
        atol=1e-6,
        rtol=0,
    )


def test_nominal_relative_states_are_global_frame_invariant():
    params = KinematicBicycleParams(
        dt=0.05, wheelbase=3.9, steering_ratio=25.0, steering_offset=0.0
    )
    action = torch.tensor(
        [[0.2, 0.1], [0.0, 0.2], [-0.1, -0.1], [0.0, 0.0]],
        dtype=torch.float32,
    )
    initial = torch.tensor([1.0, 2.0, 0.3, 8.0, 0.1])
    rotation = 1.1
    translation = torch.tensor([20.0, -7.0])
    cos_rotation = torch.cos(torch.tensor(rotation))
    sin_rotation = torch.sin(torch.tensor(rotation))
    transformed_xy = torch.stack(
        (
            initial[0] * cos_rotation - initial[1] * sin_rotation,
            initial[0] * sin_rotation + initial[1] * cos_rotation,
        )
    ) + translation
    transformed = initial.clone()
    transformed[:2] = transformed_xy
    transformed[2] = initial[2] + rotation

    states, transitions = rollout_nominal_kinematic_consistent(
        initial, action, params
    )
    transformed_states, transformed_transitions = (
        rollout_nominal_kinematic_consistent(transformed, action, params)
    )
    relative = states_relative_to_initial_body(initial, states)
    transformed_relative = states_relative_to_initial_body(
        transformed, transformed_states
    )

    torch.testing.assert_close(transformed_transitions, transitions)
    torch.testing.assert_close(transformed_relative, relative, atol=2e-6, rtol=0)


class _FakeMujocoDataset:
    def __init__(self):
        self.data = torch.zeros(1, 301, 9)
        self.history = torch.zeros(1, 251, 8)
        self.y = torch.zeros(1, 50, 6)
        self.mask = torch.zeros(1, 50)
        self.data[0, :, 3] = 10.0
        self.data[0, :, 6] = torch.arange(301, dtype=torch.float32)
        self.data[0, :, 7] = 1000.0 + torch.arange(301, dtype=torch.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.history[idx], SimpleNamespace(), self.y[idx], self.mask[idx]


def test_nuplan_action_alignment_uses_current_throttle_and_next_steer():
    view = NuPlanKinematicResidualDataset(
        _FakeMujocoDataset(),
        model_history_length=250,
        prediction_length=50,
        params=KinematicBicycleParams(),
        steer_shift=1,
    )
    sample = view[0]
    assert sample["action"][0, 0].item() == 250.0
    assert sample["action"][0, 1].item() == 1251.0
    assert sample["action"][-1, 0].item() == 299.0
    assert sample["action"][-1, 1].item() == 1300.0
    assert sample["direct_target"].shape == (50, 4)
    assert sample["residual_target"].shape == (50, 4)
    assert sample["nominal_state_rel"].shape == (50, 5)
    assert sample["nominal_transition"].shape == (50, 4)
    torch.testing.assert_close(
        sample["base_transition"],
        kinematic_consistent_transition(
            view.dataset.data[0, 250:300, [0, 1, 2, 3, 5]],
            sample["action"],
            view.params,
        ),
    )
    expected_nominal_state, expected_nominal_transition = (
        rollout_nominal_kinematic_consistent(
            sample["initial_state"], sample["action"], view.params
        )
    )
    torch.testing.assert_close(
        sample["nominal_state_rel"],
        states_relative_to_initial_body(
            sample["initial_state"], expected_nominal_state
        ),
    )
    torch.testing.assert_close(
        sample["nominal_transition"], expected_nominal_transition
    )


def test_dataset_nominal_query_does_not_use_future_truth():
    original_dataset = _FakeMujocoDataset()
    changed_dataset = _FakeMujocoDataset()
    changed_dataset.data[0, 251:, :6] = torch.randn_like(
        changed_dataset.data[0, 251:, :6]
    ) * 1000.0
    params = KinematicBicycleParams()
    original = NuPlanKinematicResidualDataset(
        original_dataset, 250, 50, params, steer_shift=1
    )[0]
    changed = NuPlanKinematicResidualDataset(
        changed_dataset, 250, 50, params, steer_shift=1
    )[0]

    assert not torch.equal(original["truth"], changed["truth"])
    torch.testing.assert_close(
        original["nominal_state_rel"], changed["nominal_state_rel"]
    )
    torch.testing.assert_close(
        original["nominal_transition"], changed["nominal_transition"]
    )


def test_rollout_metric_errors_use_position_norm_and_wrapped_yaw():
    truth = torch.zeros(1, 2, 5)
    prediction = truth.clone()
    prediction[0, 0] = torch.tensor([3.0, 4.0, 2 * torch.pi - 0.2, 2.0, -3.0])
    prediction[0, 1] = torch.tensor([0.0, 2.0, -2 * torch.pi + 0.1, -4.0, 5.0])
    errors = rollout_metric_squared_errors(prediction, truth, (1, 2))
    expected = torch.tensor([[[25.0, 0.04, 4.0, 9.0], [4.0, 0.01, 16.0, 25.0]]])
    torch.testing.assert_close(errors, expected, atol=1e-5, rtol=0)


def test_multi_horizon_rollout_loss_is_normalized_and_differentiable():
    params = KinematicBicycleParams(
        dt=0.05, wheelbase=3.9, steering_ratio=25.0, steering_offset=0.0
    )
    initial = torch.tensor([[0.0, 0.0, 0.0, 10.0, 0.0]])
    action = torch.zeros(1, 2, 2)
    residual = torch.zeros(1, 2, 4, requires_grad=True)
    prediction = rollout_kinematic_residual_consistent(
        initial, action, residual, params
    )
    truth = prediction.detach().clone()
    truth[..., 0] += torch.tensor([[0.1, 0.2]])
    scales = torch.tensor([[0.1, 0.01, 1.0, 0.1], [0.2, 0.01, 1.0, 0.1]])
    loss = multi_horizon_rollout_loss(
        prediction, truth, scales, horizons=(1, 2)
    )
    torch.testing.assert_close(loss, torch.tensor(0.25))
    loss.backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()
    assert torch.count_nonzero(residual.grad).item() > 0

    with pytest.raises(ValueError, match="shape"):
        multi_horizon_rollout_loss(
            prediction.detach(), truth, torch.ones(2, 3), horizons=(1, 2)
        )
    with pytest.raises(ValueError, match="strictly positive"):
        multi_horizon_rollout_loss(
            prediction.detach(), truth, torch.zeros(2, 4), horizons=(1, 2)
        )

"""Kinematic baseline and observable residual helpers for nuPlan data.

The raw AnyCar dataset state is ``[x, y, yaw, vx, vy, yawrate]``.  This
module deliberately exposes the observable subset
``[x, y, yaw, vx, yawrate]`` so the same protocol can later be used with
real data that does not provide lateral velocity.
"""

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


OBSERVABLE_STATE_INDICES = (0, 1, 2, 3, 5)
OBSERVABLE_STATE_NAMES = ("x", "y", "yaw", "vx", "yawrate")
OBSERVABLE_DELTA_NAMES = ("dx_body", "dy_body", "dyaw", "dvx", "dyawrate")
CONSISTENT_TRANSITION_INDICES = (0, 1, 3, 4)
CONSISTENT_TRANSITION_NAMES = ("dx_body", "dy_body", "dvx", "dyawrate")
ROLLOUT_METRIC_NAMES = ("position", "yaw", "vx", "yawrate")


@dataclass(frozen=True)
class KinematicBicycleParams:
    dt: float = 0.05
    wheelbase: float = 3.9
    steering_ratio: float = 25.0
    steering_offset: float = math.radians(5.0)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def nominal_front_wheel_angle(
    steer_command: torch.Tensor, params: KinematicBicycleParams
) -> torch.Tensor:
    """Convert logged steering-wheel command to nominal front-wheel angle."""
    return (steer_command - params.steering_offset) / params.steering_ratio


def kinematic_delta(
    current_state: torch.Tensor,
    action: torch.Tensor,
    params: KinematicBicycleParams,
) -> torch.Tensor:
    """Return one-step observable delta in the current vehicle frame.

    Args:
        current_state: ``[..., 5]`` as ``[x, y, yaw, vx, yawrate]``.
        action: ``[..., 2]`` as aligned ``[acceleration, steering_command]``.

    Returns:
        ``[..., 5]`` as ``[dx_body, dy_body, dyaw, dvx, dyawrate]``.
    """
    vx = current_state[..., 3]
    yawrate = current_state[..., 4]
    acceleration = action[..., 0]
    delta = nominal_front_wheel_angle(action[..., 1], params)

    dvx = acceleration * params.dt
    vx_next = vx + dvx
    vx_mid = 0.5 * (vx + vx_next)
    yawrate_next = vx_mid * torch.tan(delta) / params.wheelbase

    dx_body = vx_mid * params.dt
    dy_body = torch.zeros_like(dx_body)
    dyaw = yawrate_next * params.dt
    dyawrate = yawrate_next - yawrate
    return torch.stack((dx_body, dy_body, dyaw, dvx, dyawrate), dim=-1)


def kinematic_consistent_transition(
    current_state: torch.Tensor,
    action: torch.Tensor,
    params: KinematicBicycleParams,
) -> torch.Tensor:
    """Return the v2 transition primitives without an independent yaw delta.

    The nuPlan generator advances yaw with the current yawrate.  The predicted
    next yawrate therefore affects yaw starting from the following transition.
    """
    delta = kinematic_delta(current_state, action, params)
    return delta[..., list(CONSISTENT_TRANSITION_INDICES)]


def apply_consistent_transition(
    state: torch.Tensor,
    transition: torch.Tensor,
    params: KinematicBicycleParams,
) -> torch.Tensor:
    """Apply ``[dx_body, dy_body, dvx, dyawrate]`` with exact yaw integration."""
    yaw = state[..., 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    dx_world = transition[..., 0] * cos_yaw - transition[..., 1] * sin_yaw
    dy_world = transition[..., 0] * sin_yaw + transition[..., 1] * cos_yaw

    next_state = state.clone()
    next_state[..., 0] = state[..., 0] + dx_world
    next_state[..., 1] = state[..., 1] + dy_world
    next_state[..., 2] = wrap_angle(state[..., 2] + state[..., 4] * params.dt)
    next_state[..., 3] = state[..., 3] + transition[..., 2]
    next_state[..., 4] = state[..., 4] + transition[..., 3]
    return next_state


def apply_observable_delta(
    state: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    """Apply a body-frame delta to observable absolute state."""
    yaw = state[..., 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    dx_world = delta[..., 0] * cos_yaw - delta[..., 1] * sin_yaw
    dy_world = delta[..., 0] * sin_yaw + delta[..., 1] * cos_yaw

    next_state = state.clone()
    next_state[..., 0] = state[..., 0] + dx_world
    next_state[..., 1] = state[..., 1] + dy_world
    next_state[..., 2] = wrap_angle(state[..., 2] + delta[..., 2])
    next_state[..., 3] = state[..., 3] + delta[..., 3]
    next_state[..., 4] = state[..., 4] + delta[..., 4]
    return next_state


def rollout_delta_sequence(
    initial_state: torch.Tensor, delta_sequence: torch.Tensor
) -> torch.Tensor:
    """Roll out already-composed observable delta predictions."""
    state = initial_state
    outputs = []
    for step in range(delta_sequence.shape[1]):
        state = apply_observable_delta(state, delta_sequence[:, step])
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def rollout_kinematic_residual(
    initial_state: torch.Tensor,
    action: torch.Tensor,
    residual: torch.Tensor,
    params: KinematicBicycleParams,
) -> torch.Tensor:
    """Recursively roll out the fixed kinematic model plus learned residual."""
    state = initial_state
    outputs = []
    for step in range(action.shape[1]):
        base_delta = kinematic_delta(state, action[:, step], params)
        state = apply_observable_delta(state, base_delta + residual[:, step])
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def rollout_consistent_transition_sequence(
    initial_state: torch.Tensor,
    transition_sequence: torch.Tensor,
    params: KinematicBicycleParams,
) -> torch.Tensor:
    """Roll out direct-v2 transition primitives with the shared integrator."""
    state = initial_state
    outputs = []
    for step in range(transition_sequence.shape[1]):
        state = apply_consistent_transition(state, transition_sequence[:, step], params)
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def rollout_nominal_kinematic_consistent(
    initial_state: torch.Tensor,
    action: torch.Tensor,
    params: KinematicBicycleParams,
):
    """Independently roll out the nominal model from current observable state.

    Only ``initial_state`` and the aligned future ``action`` sequence are used.
    The function deliberately does not accept future ground-truth states, so the
    returned trajectory is available under the same protocol in training and
    inference.

    Args:
        initial_state: ``[..., 5]`` observable current state.
        action: ``[..., horizon, 2]`` aligned future actions.

    Returns:
        A pair ``(states, transitions)`` with shapes ``[..., horizon, 5]`` and
        ``[..., horizon, 4]`` respectively.
    """
    state = initial_state
    states = []
    transitions = []
    for step in range(action.shape[-2]):
        transition = kinematic_consistent_transition(
            state, action.select(dim=-2, index=step), params
        )
        state = apply_consistent_transition(state, transition, params)
        transitions.append(transition)
        states.append(state)
    return torch.stack(states, dim=-2), torch.stack(transitions, dim=-2)


def states_relative_to_initial_body(
    initial_state: torch.Tensor, states: torch.Tensor
) -> torch.Tensor:
    """Express observable states relative to the initial vehicle frame.

    Position and yaw become invariant to a global rigid transform.  Longitudinal
    speed and yawrate remain physical absolute quantities because they are the
    dynamic state components needed by the residual query.
    """
    initial = initial_state.unsqueeze(-2)
    dx_world = states[..., 0] - initial[..., 0]
    dy_world = states[..., 1] - initial[..., 1]
    initial_yaw = initial[..., 2]
    cos_yaw = torch.cos(initial_yaw)
    sin_yaw = torch.sin(initial_yaw)
    x_body = dx_world * cos_yaw + dy_world * sin_yaw
    y_body = -dx_world * sin_yaw + dy_world * cos_yaw
    yaw_relative = wrap_angle(states[..., 2] - initial_yaw)
    return torch.stack(
        (x_body, y_body, yaw_relative, states[..., 3], states[..., 4]), dim=-1
    )


def rollout_metric_squared_errors(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    horizons,
) -> torch.Tensor:
    """Return physical squared rollout errors at selected one-based horizons.

    The last dimension is ordered as ``[position, yaw, vx, yawrate]``.
    Position is the squared 2-D Euclidean error and yaw uses the wrapped angle
    difference.  Inputs may be expressed in either the global frame or the same
    relative body frame because all four errors are frame invariant.
    """
    if prediction.shape != truth.shape or prediction.shape[-1] != 5:
        raise ValueError(
            "prediction and truth must have the same [..., horizon, 5] shape"
        )
    horizon_indices = tuple(int(horizon) - 1 for horizon in horizons)
    if not horizon_indices:
        raise ValueError("at least one rollout horizon is required")
    if min(horizon_indices) < 0 or max(horizon_indices) >= prediction.shape[-2]:
        raise ValueError(
            f"rollout horizons must be within [1, {prediction.shape[-2]}]"
        )
    index = torch.tensor(
        horizon_indices, dtype=torch.long, device=prediction.device
    )
    diff = prediction.index_select(-2, index) - truth.index_select(-2, index)
    yaw_diff = wrap_angle(diff[..., 2])
    return torch.stack(
        (
            diff[..., 0].square() + diff[..., 1].square(),
            yaw_diff.square(),
            diff[..., 3].square(),
            diff[..., 4].square(),
        ),
        dim=-1,
    )


def multi_horizon_rollout_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    scales: torch.Tensor,
    horizons,
) -> torch.Tensor:
    """Return a dimensionless mean rollout MSE over horizons and metrics."""
    errors = rollout_metric_squared_errors(prediction, truth, horizons)
    expected_shape = (errors.shape[-2], len(ROLLOUT_METRIC_NAMES))
    if tuple(scales.shape) != expected_shape:
        raise ValueError(
            f"rollout scales must have shape {expected_shape}, got {tuple(scales.shape)}"
        )
    if torch.any(scales <= 0):
        raise ValueError("rollout scales must be strictly positive")
    scale = scales.to(device=prediction.device, dtype=prediction.dtype)
    return torch.mean(errors / scale.square())


def rollout_kinematic_residual_consistent(
    initial_state: torch.Tensor,
    action: torch.Tensor,
    residual: torch.Tensor,
    params: KinematicBicycleParams,
) -> torch.Tensor:
    """Roll out kinematic plus residual v2 with the shared integrator."""
    state = initial_state
    outputs = []
    for step in range(action.shape[1]):
        base_transition = kinematic_consistent_transition(
            state, action[:, step], params
        )
        state = apply_consistent_transition(
            state, base_transition + residual[:, step], params
        )
        outputs.append(state)
    return torch.stack(outputs, dim=1)


class NuPlanKinematicResidualDataset(Dataset):
    """Observable direct/residual view over ``MujocoDataset``.

    ``MujocoDataset`` is constructed with ``model_history_length + 1`` history
    frames.  For the first future transition the current absolute state is at
    raw index ``model_history_length`` and the target state is at the next
    index.  The generator records acceleration for the next longitudinal
    transition, but records steering after applying it to the just-produced
    state.  Consequently the aligned future control is
    ``[throttle[t], steer[t + 1]]`` by default.
    """

    def __init__(
        self,
        dataset,
        model_history_length: int,
        prediction_length: int,
        params: KinematicBicycleParams,
        steer_shift: int = 1,
    ):
        self.dataset = dataset
        self.model_history_length = model_history_length
        self.prediction_length = prediction_length
        self.params = params
        self.steer_shift = steer_shift
        self.state_indices = torch.tensor(OBSERVABLE_STATE_INDICES, dtype=torch.long)

        required_last = model_history_length + prediction_length - 1 + max(steer_shift, 0)
        if required_last >= dataset.data.shape[1]:
            raise ValueError(
                "Dataset sequence is too short for requested prediction length "
                f"and steer_shift={steer_shift}: {dataset.data.shape[1]} frames"
            )

        start = self.model_history_length
        stop = start + self.prediction_length
        all_actions = torch.stack(
            (
                self.dataset.data[:, start:stop, 6],
                self.dataset.data[
                    :,
                    start + self.steer_shift : stop + self.steer_shift,
                    7,
                ],
            ),
            dim=-1,
        )
        all_initial_states = self.dataset.data[:, start, :6][
            :, self.state_indices
        ]
        nominal_state, self.nominal_transition = (
            rollout_nominal_kinematic_consistent(
                all_initial_states, all_actions, self.params
            )
        )
        self.nominal_state_rel = states_relative_to_initial_body(
            all_initial_states, nominal_state
        )

    def __len__(self):
        return len(self.dataset)

    def _aligned_action(self, idx: int) -> torch.Tensor:
        start = self.model_history_length
        stop = start + self.prediction_length
        throttle = self.dataset.data[idx, start:stop, 6]
        steer_start = start + self.steer_shift
        steer = self.dataset.data[
            idx, steer_start : steer_start + self.prediction_length, 7
        ]
        return torch.stack((throttle, steer), dim=-1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_history, _, raw_target, mask = self.dataset[idx]
        history_state = raw_history[:, self.state_indices]
        history = torch.cat((history_state, raw_history[:, 6:8]), dim=-1)
        direct_delta = raw_target[:, self.state_indices]
        direct_target = direct_delta[:, list(CONSISTENT_TRANSITION_INDICES)]
        action = self._aligned_action(idx)

        start = self.model_history_length
        current_sequence = self.dataset.data[
            idx, start : start + self.prediction_length, :6
        ][:, self.state_indices]
        base_transition = kinematic_consistent_transition(
            current_sequence, action, self.params
        )
        residual_target = direct_target - base_transition

        initial_state = self.dataset.data[idx, start, :6][self.state_indices]
        truth = self.dataset.data[
            idx, start + 1 : start + 1 + self.prediction_length, :6
        ][:, self.state_indices]
        current_context = torch.stack(
            (
                initial_state[3],
                initial_state[4],
                self.dataset.data[idx, start, 6],
                self.dataset.data[idx, start, 7],
            )
        )
        return {
            "history": history,
            "action": action,
            "direct_target": direct_target,
            "base_transition": base_transition,
            "residual_target": residual_target,
            "initial_state": initial_state,
            "nominal_state_rel": self.nominal_state_rel[idx],
            "nominal_transition": self.nominal_transition[idx],
            "truth": truth,
            "current_context": current_context,
            "mask": mask,
        }


def tensor_stats(values: torch.Tensor, eps: float = 1e-6):
    mean = values.mean(dim=0)
    std = values.std(dim=0).clamp_min(eps)
    return mean, std

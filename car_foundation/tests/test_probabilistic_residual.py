import math

import pytest
import torch

from car_foundation.probabilistic_residual import (
    channel_temperature,
    gaussian_nll,
    horizon_channel_temperature,
    inverse_softplus,
    positive_scale,
    student_t_nll,
    student_t_standard_deviation,
)


def test_inverse_softplus_round_trip():
    value = torch.tensor([0.001, 0.1, 1.0, 20.0])
    torch.testing.assert_close(torch.nn.functional.softplus(inverse_softplus(value)), value)


def test_positive_scale_has_floor_and_gradient():
    raw = torch.tensor([-100.0, 0.0, 10.0], requires_grad=True)
    sigma = positive_scale(raw, floor=1e-3)
    assert torch.all(sigma >= 1e-3)
    sigma.sum().backward()
    assert torch.all(torch.isfinite(raw.grad))


def test_gaussian_nll_matches_standard_normal_constant():
    error = torch.zeros(2, 3, 4)
    sigma = torch.ones_like(error)
    actual = gaussian_nll(error, sigma, include_constant=True)
    assert actual.item() == pytest.approx(0.5 * math.log(2.0 * math.pi))


def test_gaussian_nll_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        gaussian_nll(torch.zeros(2, 4), torch.ones(2, 3))
    with pytest.raises(ValueError):
        gaussian_nll(torch.zeros(2, 4), torch.zeros(2, 4))
    with pytest.raises(ValueError):
        gaussian_nll(
            torch.zeros(2, 4),
            torch.ones(2, 4),
            weights=torch.ones(3),
        )


def test_channel_temperature_recovers_multiplicative_error_scale():
    sigma = torch.ones(2, 3, 2)
    pattern = torch.tensor([-1.0, 1.0]).view(2, 1, 1)
    error = (
        pattern * torch.tensor([2.0, 0.5]).view(1, 1, 2)
    ).expand_as(sigma)
    temperature = channel_temperature(error, sigma)
    torch.testing.assert_close(temperature, torch.tensor([2.0, 0.5]))


def test_horizon_channel_temperature_keeps_horizon_axis():
    sigma = torch.ones(2, 3, 2)
    scale = torch.tensor(
        [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]
    ).view(1, 3, 2)
    sign = torch.tensor([-1.0, 1.0]).view(2, 1, 1)
    error = sign * scale
    temperature = horizon_channel_temperature(error, sigma)
    torch.testing.assert_close(temperature, scale.squeeze(0))


def test_horizon_channel_temperature_requires_three_dimensions():
    with pytest.raises(ValueError):
        horizon_channel_temperature(torch.ones(2, 4), torch.ones(2, 4))


def test_student_t_nll_matches_torch_distribution():
    error = torch.tensor(
        [[[-1.0, 0.0], [0.5, 2.0]]],
        dtype=torch.float64,
    )
    scale = torch.tensor([0.7, 1.5], dtype=torch.float64)
    degrees_of_freedom = torch.tensor([3.0, 10.0], dtype=torch.float64)
    expected = -torch.distributions.StudentT(
        degrees_of_freedom,
        loc=torch.zeros_like(degrees_of_freedom),
        scale=scale,
    ).log_prob(error).mean()
    actual = student_t_nll(error, scale, degrees_of_freedom)
    torch.testing.assert_close(actual, expected)


def test_student_t_nll_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        student_t_nll(
            torch.ones(2, 3),
            torch.zeros(2, 3),
            torch.tensor(3.0),
        )
    with pytest.raises(ValueError):
        student_t_nll(
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.tensor(0.0),
        )
    with pytest.raises(ValueError):
        student_t_nll(
            torch.ones(2, 3),
            torch.ones(2, 2),
            torch.tensor(3.0),
        )


def test_student_t_standard_deviation():
    scale = torch.tensor([1.0, 2.0])
    degrees_of_freedom = torch.tensor([3.0, 10.0])
    actual = student_t_standard_deviation(scale, degrees_of_freedom)
    expected = scale * torch.sqrt(
        degrees_of_freedom / (degrees_of_freedom - 2.0)
    )
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        student_t_standard_deviation(
            torch.ones(1),
            torch.tensor([2.0]),
        )

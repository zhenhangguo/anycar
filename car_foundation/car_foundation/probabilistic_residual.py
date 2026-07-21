"""Probability helpers for frozen-mean kinematic residual predictors."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus for strictly positive values."""
    if torch.any(value <= 0):
        raise ValueError("inverse_softplus expects strictly positive values")
    return value + torch.log(-torch.expm1(-value))


def positive_scale(raw_scale: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
    """Map unconstrained values to finite standard deviations."""
    if floor <= 0:
        raise ValueError("floor must be strictly positive")
    return floor + F.softplus(raw_scale)


def gaussian_nll(
    error: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor | None = None,
    include_constant: bool = False,
) -> torch.Tensor:
    """Mean diagonal-Gaussian NLL.

    ``error`` and ``sigma`` must have identical shapes. Channel weights are
    applied to the final per-element NLL and broadcast over leading axes.
    """
    if error.shape != sigma.shape:
        raise ValueError("error and sigma must have identical shapes")
    if torch.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive")
    loss = 0.5 * ((error / sigma).square() + 2.0 * torch.log(sigma))
    if include_constant:
        loss = loss + 0.5 * math.log(2.0 * math.pi)
    if weights is not None:
        if weights.ndim != 1 or weights.shape[0] != error.shape[-1]:
            raise ValueError("weights must contain one value per output channel")
        loss = loss * weights.to(device=loss.device, dtype=loss.dtype)
    return loss.mean()


def student_t_nll(
    error: torch.Tensor,
    scale: torch.Tensor,
    degrees_of_freedom: torch.Tensor,
) -> torch.Tensor:
    """Mean zero-location Student-t NLL with broadcastable parameters."""
    if torch.any(scale <= 0):
        raise ValueError("scale must be strictly positive")
    if torch.any(degrees_of_freedom <= 0):
        raise ValueError("degrees_of_freedom must be strictly positive")
    try:
        torch.broadcast_shapes(
            error.shape,
            scale.shape,
            degrees_of_freedom.shape,
        )
    except RuntimeError as error_shape:
        raise ValueError(
            "error, scale, and degrees_of_freedom must be broadcastable"
        ) from error_shape
    half_df = 0.5 * degrees_of_freedom
    loss = (
        torch.log(scale)
        + 0.5
        * (
            torch.log(degrees_of_freedom)
            + math.log(math.pi)
        )
        + torch.lgamma(half_df)
        - torch.lgamma(half_df + 0.5)
        + (half_df + 0.5)
        * torch.log1p(
            (error / scale).square() / degrees_of_freedom
        )
    )
    return loss.mean()


def student_t_standard_deviation(
    scale: torch.Tensor,
    degrees_of_freedom: torch.Tensor,
) -> torch.Tensor:
    """Convert Student-t scale to standard deviation for df greater than two."""
    if torch.any(scale <= 0):
        raise ValueError("scale must be strictly positive")
    if torch.any(degrees_of_freedom <= 2):
        raise ValueError(
            "degrees_of_freedom must be greater than two for finite variance"
        )
    return scale * torch.sqrt(
        degrees_of_freedom / (degrees_of_freedom - 2.0)
    )


def channel_temperature(error: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Closed-form per-channel Gaussian temperature on a calibration set."""
    if error.shape != sigma.shape:
        raise ValueError("error and sigma must have identical shapes")
    reduce_dims = tuple(range(error.ndim - 1))
    return torch.sqrt(torch.mean((error / sigma).square(), dim=reduce_dims))


def horizon_channel_temperature(
    error: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Closed-form temperature for every horizon and output channel."""
    if error.shape != sigma.shape:
        raise ValueError("error and sigma must have identical shapes")
    if error.ndim != 3:
        raise ValueError("Expected [batch, horizon, channel] tensors")
    return torch.sqrt(torch.mean((error / sigma).square(), dim=0))


class FrozenMeanGaussianResidual(nn.Module):
    """Attach a trainable diagonal-Gaussian scale head to a frozen Query model.

    The deterministic output head and the sigma head consume the same decoder
    hidden sequence. The wrapped mean model is always kept in evaluation mode,
    including while the sigma head is trained.
    """

    def __init__(
        self,
        mean_model: nn.Module,
        output_dim: int = 4,
        sigma_floor: float = 1e-3,
    ):
        super().__init__()
        self.mean_model = mean_model
        self.sigma_floor = sigma_floor
        self.sigma_head = nn.Linear(mean_model.latent_dim, output_dim).to(
            mean_model.device
        )
        for parameter in self.mean_model.parameters():
            parameter.requires_grad_(False)
        self.mean_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.mean_model.eval()
        return self

    def initialize_constant_sigma(self, sigma: torch.Tensor) -> None:
        """Initialize the head to an input-independent normalized sigma."""
        sigma = sigma.to(
            device=self.sigma_head.bias.device,
            dtype=self.sigma_head.bias.dtype,
        )
        if sigma.shape != self.sigma_head.bias.shape:
            raise ValueError(
                f"Expected sigma shape {tuple(self.sigma_head.bias.shape)}, "
                f"got {tuple(sigma.shape)}"
            )
        adjusted = sigma - self.sigma_floor
        if torch.any(adjusted <= 0):
            raise ValueError("Initial sigma must be greater than sigma_floor")
        with torch.no_grad():
            self.sigma_head.weight.zero_()
            self.sigma_head.bias.copy_(inverse_softplus(adjusted))

    def forward(
        self,
        history: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
        nominal_state: torch.Tensor | None = None,
        nominal_transition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.mean_model
        with torch.no_grad():
            history_emb = model.position_encoding["history"](
                model._build_history_emb(history)
            )
            action_embedding = model._build_action_emb(
                history,
                action,
                context,
                nominal_state,
                nominal_transition,
            )
            action_emb = model.position_encoding["action"](action_embedding)
            hidden = model.transformer_decoder(
                tgt=action_emb,
                memory=history_emb,
                tgt_mask=model.tgt_mask,
                tgt_key_padding_mask=(
                    mask.to(model.device) if mask is not None else None
                ),
            )
            mean = model.embedding["output"](hidden)
        sigma = positive_scale(self.sigma_head(hidden.detach()), self.sigma_floor)
        return mean, sigma

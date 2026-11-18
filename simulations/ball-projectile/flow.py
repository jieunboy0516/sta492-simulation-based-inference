"""A one-dimensional conditional spline flow for q(angle | measured distance).

We borrow only the spline calculation from nflows. The small conditioning
networks, change of variables, angle bounds, and sampling are written here.
All angle densities are per DEGREE, matching the prior and the plots.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)


class ConditionalSpline(nn.Module):
    """Predict a monotone spline's shape from the observed distance.

    There are K widths, K heights, and K-1 interior derivatives. nflows makes
    widths/heights positive using softmax and derivatives positive using
    softplus. Positivity makes the spline invertible.
    """

    def __init__(self, num_bins=16, hidden_size=32):
        super().__init__()
        self.num_bins = num_bins
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3 * num_bins - 1),
        )
        # Equal bin widths/heights and unit derivatives give the identity map.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[2 * num_bins :] = math.log(math.expm1(1 - 1e-3))

    def forward(self, values, context, inverse=False):
        parameters = self.network(context)
        widths = parameters[:, : self.num_bins]
        heights = parameters[:, self.num_bins : 2 * self.num_bins]
        derivatives = parameters[:, 2 * self.num_bins :]
        output, log_determinant = unconstrained_rational_quadratic_spline(
            inputs=values[:, 0],
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=inverse,
            tails="linear",
            tail_bound=5.0,
        )
        return output[:, None], log_determinant


class ConditionalAngleFlow(nn.Module):
    """A Gaussian base, three conditional splines, and a bounded angle map.

    log_prob(angle_deg, distance_m): angles have shape (B, 1); distances have
    shape (B, 1), or a scalar for one shared observation. Returns shape (B,).

    sample(num_samples, distance_m): one distance in, shape (num_samples, 1) out.
    """

    def __init__(self, num_layers=3, num_bins=16, hidden_size=32):
        super().__init__()
        self.register_buffer("distance_mean", torch.zeros(1))
        self.register_buffer("distance_std", torch.ones(1))
        self.layers = nn.ModuleList(
            [ConditionalSpline(num_bins, hidden_size) for _ in range(num_layers)]
        )

    @torch.no_grad()
    def fit_standardization(self, training_distances):
        """Use only training observations, never validation or test observations."""
        self.distance_mean.copy_(training_distances.mean(0))
        self.distance_std.copy_(training_distances.std(0, unbiased=False).clamp_min(1e-6))

    def _raw_distance(self, distance_m, batch_size):
        distance = torch.as_tensor(
            distance_m, dtype=self.distance_mean.dtype, device=self.distance_mean.device
        )
        if distance.ndim == 0:
            distance = distance.reshape(1, 1)
        elif distance.ndim == 1:
            distance = distance[:, None]
        if distance.ndim != 2 or distance.shape[1] != 1:
            raise ValueError("Distances must have shape (B, 1), or be a scalar.")
        if distance.shape[0] == 1:
            distance = distance.expand(batch_size, -1)
        elif distance.shape[0] != batch_size:
            raise ValueError("Angle and distance batch sizes must agree.")
        if not torch.isfinite(distance).all():
            raise ValueError("Distances must be finite.")
        return distance

    def _context(self, distance_m, batch_size):
        distance = self._raw_distance(distance_m, batch_size)
        return (distance - self.distance_mean) / self.distance_std

    def to_base(self, angle_deg, distance_m):
        """Angle -> logit(angle/90) -> Gaussian coordinate; INVERSE log-Jacobian."""
        if angle_deg.ndim != 2 or angle_deg.shape[1] != 1:
            raise ValueError("Angles must have shape (B, 1).")
        if not ((angle_deg > 0) & (angle_deg < 90)).all():
            raise ValueError("to_base requires angles strictly between 0 and 90.")
        context = self._context(distance_m, len(angle_deg))
        fraction = angle_deg / 90.0
        values = torch.log(fraction) - torch.log1p(-fraction)
        log_determinant = -(
            math.log(90.0) + torch.log(fraction) + torch.log1p(-fraction)
        ).sum(1)
        # Our spline forward direction is data -> base, as in density estimation.
        for layer in self.layers:
            values, log_det = layer(values, context)
            log_determinant = log_determinant + log_det
        return values, log_determinant

    def to_angles(self, base_samples, distance_m):
        """Gaussian coordinate -> angle; return the FORWARD log-Jacobian."""
        context = self._context(distance_m, len(base_samples))
        values = base_samples
        log_determinant = base_samples.new_zeros(len(base_samples))
        for layer in reversed(self.layers):
            values, log_det = layer(values, context, inverse=True)
            log_determinant = log_determinant + log_det
        angles = 90.0 * torch.sigmoid(values)
        box_log_det = (
            math.log(90.0) + F.logsigmoid(values) + F.logsigmoid(-values)
        ).sum(1)
        return angles, log_determinant + box_log_det

    def log_prob(self, angle_deg, distance_m):
        """Base Gaussian log-density plus the angle-to-base log-Jacobian."""
        if angle_deg.ndim != 2 or angle_deg.shape[1] != 1:
            raise ValueError("Angles must have shape (B, 1).")
        distance = self._raw_distance(distance_m, len(angle_deg))
        inside = ((angle_deg > 0) & (angle_deg < 90)).all(1)
        result = angle_deg.new_full((len(angle_deg),), -torch.inf)
        if inside.any():
            z, log_det = self.to_base(angle_deg[inside], distance[inside])
            base_log_density = -0.5 * (z.square() + math.log(2 * math.pi)).sum(1)
            result[inside] = base_log_density + log_det
        return result

    @torch.no_grad()
    def sample(self, num_samples, distance_m):
        """Sample one posterior without retraining or evaluating a likelihood."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        distance = self._raw_distance(distance_m, 1)  # require one observation
        z = torch.randn(
            num_samples, 1, dtype=self.distance_mean.dtype, device=self.distance_mean.device
        )
        angles, _ = self.to_angles(z, distance)
        return angles

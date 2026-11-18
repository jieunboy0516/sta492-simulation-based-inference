"""A small conditional normalizing flow, written to be read alongside the notebook.

Sampling goes from left to right:
    Gaussian z -> six invertible coupling layers -> u -> bounded parameters theta.

Density evaluation goes in the opposite direction. Each transformation returns
its log absolute Jacobian determinant, so we can keep track of changing density.
There is no simulator or likelihood in this file.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class AffineCoupling(nn.Module):
    """Keep one coordinate; scale and shift the other using it and the data."""

    def __init__(self, observation_size, hidden_size=64, keep_coordinate=0):
        super().__init__()
        self.keep_coordinate = keep_coordinate
        self.change_coordinate = 1 - keep_coordinate
        self.network = nn.Sequential(
            nn.Linear(1 + observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),  # outputs: raw log-scale and shift
        )
        # Begin as the identity map. Training then learns useful departures.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _scale_and_shift(self, values, observation):
        kept_value = values[:, self.keep_coordinate : self.keep_coordinate + 1]
        network_input = torch.cat([kept_value, observation], dim=1)
        raw_log_scale, shift = self.network(network_input).unbind(dim=1)
        # Avoid enormous exp(log_scale) values while keeping the map invertible.
        log_scale = 1.5 * torch.tanh(raw_log_scale)
        return log_scale, shift

    def forward(self, values, observation):
        """v_changed = u_changed * exp(s) + shift; log|det J| = s."""
        log_scale, shift = self._scale_and_shift(values, observation)
        transformed = values.clone()
        transformed[:, self.change_coordinate] = (
            values[:, self.change_coordinate] * torch.exp(log_scale) + shift
        )
        return transformed, log_scale

    def inverse(self, values, observation):
        """u_changed = (v_changed - shift) * exp(-s); log|det J| = -s."""
        # The kept coordinate did not change, so it gives the same s and shift.
        log_scale, shift = self._scale_and_shift(values, observation)
        transformed = values.clone()
        transformed[:, self.change_coordinate] = (
            values[:, self.change_coordinate] - shift
        ) * torch.exp(-log_scale)
        return transformed, -log_scale


class ConditionalFlow(nn.Module):
    """Learn q(theta | observation) for two parameters in a rectangular box.

    log_prob(theta, observation): theta has shape (batch, 2). Observation has
    shape (batch, observation_size), or (observation_size,) for one shared dataset.
    The result has shape (batch,).

    sample(num_samples, observation): one observed dataset in, an array with
    shape (num_samples, 2) out. These are draws from the learned posterior.
    """

    def __init__(
        self,
        observation_size=20,
        lower=(-1.0, -2.0),
        upper=(1.0, 2.0),
        num_layers=6,
        hidden_size=64,
    ):
        super().__init__()
        self.observation_size = observation_size
        self.register_buffer("lower", torch.tensor(lower, dtype=torch.float32))
        self.register_buffer("upper", torch.tensor(upper, dtype=torch.float32))
        self.register_buffer("observation_mean", torch.zeros(observation_size))
        self.register_buffer("observation_std", torch.ones(observation_size))
        if (
            self.lower.shape != (2,)
            or self.upper.shape != (2,)
            or not torch.all(self.upper > self.lower)
        ):
            raise ValueError("Supply two lower bounds and two larger upper bounds.")
        self.layers = nn.ModuleList(
            [
                AffineCoupling(observation_size, hidden_size, keep_coordinate=i % 2)
                for i in range(num_layers)
            ]
        )

    @torch.no_grad()
    def fit_observation_standardization(self, training_observations):
        """Call once with TRAINING data; validation/observed data stay unseen."""
        self.observation_mean.copy_(training_observations.mean(dim=0))
        self.observation_std.copy_(
            training_observations.std(dim=0, unbiased=False).clamp_min(1e-6)
        )

    def _expand_observation(self, observation, batch_size):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[1] != self.observation_size:
            raise ValueError("Observation must have shape (batch, observation_size).")
        if observation.shape[0] == 1:
            observation = observation.expand(batch_size, -1)
        elif observation.shape[0] != batch_size:
            raise ValueError("Parameter and observation batch sizes must agree.")
        return observation

    def _context(self, observation, batch_size):
        observation = self._expand_observation(observation, batch_size)
        return (observation - self.observation_mean) / self.observation_std

    def to_parameters(self, base_samples, observation):
        """Map Gaussian samples to theta, returning the FORWARD log-Jacobian."""
        context = self._context(observation, len(base_samples))
        values = base_samples
        log_determinant = base_samples.new_zeros(len(base_samples))
        for layer in self.layers:
            values, layer_log_determinant = layer(values, context)
            log_determinant = log_determinant + layer_log_determinant

        # theta_j = lower_j + width_j * sigmoid(u_j).
        width = self.upper - self.lower
        theta = self.lower + width * torch.sigmoid(values)
        # log(sigmoid'(u)) = log(sigmoid(u)) + log(sigmoid(-u)).
        # logsigmoid is more stable than taking log(sigmoid(u)) directly.
        box_log_determinant = (
            torch.log(width) + F.logsigmoid(values) + F.logsigmoid(-values)
        ).sum(dim=1)
        return theta, log_determinant + box_log_determinant

    def to_base(self, theta, observation):
        """Map INTERIOR parameter values back to z; return INVERSE log-Jacobian."""
        if theta.ndim != 2 or theta.shape[1] != 2:
            raise ValueError("theta must have shape (batch, 2).")
        if not torch.all((theta > self.lower) & (theta < self.upper)):
            raise ValueError("to_base requires parameters strictly inside the bounds.")
        context = self._context(observation, len(theta))
        width = self.upper - self.lower
        fraction = (theta - self.lower) / width
        values = torch.log(fraction) - torch.log1p(-fraction)  # inverse sigmoid
        log_determinant = -(
            torch.log(width) + torch.log(fraction) + torch.log1p(-fraction)
        ).sum(dim=1)
        for layer in reversed(self.layers):
            values, layer_log_determinant = layer.inverse(values, context)
            log_determinant = log_determinant + layer_log_determinant
        return values, log_determinant

    def log_prob(self, theta, observation):
        """log q(theta | data) = log Normal(z) + log|det(dz/dtheta)|."""
        if theta.ndim != 2 or theta.shape[1] != 2:
            raise ValueError("theta must have shape (batch, 2).")
        # Endpoints have probability zero; choose density zero there as well.
        inside = ((theta > self.lower) & (theta < self.upper)).all(dim=1)
        log_density = theta.new_full((len(theta),), -torch.inf)
        raw_observation = self._expand_observation(observation, len(theta))
        if inside.any():
            # Select matching parameter/data rows, then evaluate only the interior.
            z, inverse_log_determinant = self.to_base(
                theta[inside], raw_observation[inside]
            )
            base_log_density = -0.5 * (z.square() + math.log(2 * math.pi)).sum(dim=1)
            log_density[inside] = base_log_density + inverse_log_determinant
        return log_density

    @torch.no_grad()
    def sample(self, num_samples, observation):
        """Draw independent Gaussian values, then transform them into parameters."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        if observation.ndim == 2 and observation.shape[0] != 1:
            raise ValueError("sample expects ONE observed dataset.")
        z = torch.randn(
            num_samples, 2, dtype=self.lower.dtype, device=self.lower.device
        )
        theta, _ = self.to_parameters(z, observation)
        return theta

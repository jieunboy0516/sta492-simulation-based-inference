"""Checks based on physics, probability, and independent numerical derivatives.

Run from this folder with: python3 -m unittest -v
"""

import io
import unittest

import numpy as np
import torch
from scipy.signal import find_peaks

from flow import ConditionalAngleFlow
from projectile import (
    SPEED, GRAVITY, WIND_RATE, MEASUREMENT_SD,
    prior_density, sample_prior, range_coefficients, trajectory,
    simulate, reference_posterior,
)


class PhysicsTests(unittest.TestCase):
    def test_landing_time_and_no_wind_range(self):
        for angle in (15.0, 45.0, 75.0):
            time, horizontal, height = trajectory(angle)
            self.assertAlmostEqual(height[-1], 0.0, places=12)
            self.assertTrue((height[1:-1] > 0).all())
            expected = SPEED**2 / GRAVITY * np.sin(2 * np.deg2rad(angle))
            self.assertAlmostEqual(horizontal[-1], expected, places=12)
            self.assertGreater(time[-1], 0)

    def test_complementary_angles_have_equal_no_wind_ranges(self):
        for angle in (10.0, 24.0, 37.0):
            first = trajectory(angle)[1][-1]
            second = trajectory(90.0 - angle)[1][-1]
            self.assertAlmostEqual(first, second, places=12)

    def test_wind_effect_matches_acceleration_integral(self):
        time, horizontal, _ = trajectory(35.0, wind_acceleration=0.7)
        no_wind = trajectory(35.0)[1][-1]
        self.assertAlmostEqual(horizontal[-1] - no_wind, 0.5 * 0.7 * time[-1]**2)

    def test_simulated_moments_match_marginal_probability_model(self):
        count = 100_000
        angles = np.full((count, 1), 35.0)
        draws = simulate(angles, np.random.default_rng(7))[:, 0]
        a, b = range_coefficients(35.0)
        mean = a + b / WIND_RATE
        variance = MEASUREMENT_SD**2 + (b / WIND_RATE)**2
        self.assertLess(abs(draws.mean() - mean), 6 * np.sqrt(variance / count))
        self.assertLess(abs(draws.var() - variance) / variance, 0.05)

    def test_prior_and_reference_posterior(self):
        draws = sample_prior(3000, np.random.default_rng(8))
        self.assertEqual(draws.shape, (3000, 1))
        self.assertTrue(((draws > 0) & (draws < 90)).all())
        angles = (np.arange(10_000) + 0.5) * 90 / 10_000
        self.assertAlmostEqual(prior_density(angles).sum() * 90 / 10_000, 1, places=6)
        coarse = reference_posterior(13.0, 2000)
        fine = reference_posterior(13.0, 4000)
        self.assertAlmostEqual(fine["density"].sum() * fine["step"], 1.0, places=12)
        self.assertLess(abs(coarse["mean_deg"] - fine["mean_deg"]), 0.001)
        self.assertLess(abs(coarse["low_angle_probability"] - fine["low_angle_probability"]), 1e-5)
        peaks, _ = find_peaks(fine["density"], prominence=0.001)
        self.assertEqual(len(peaks), 2)
        self.assertLess(fine["angle_deg"][peaks[0]], 45)
        self.assertGreater(fine["angle_deg"][peaks[1]], 45)


class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.model = ConditionalAngleFlow(hidden_size=8).double()
        # Make the maps nontrivial so inversion/Jacobian tests exercise splines.
        for layer in self.model.layers:
            torch.nn.init.normal_(layer.network[-1].weight, std=0.03)

    def test_round_trip_and_jacobian_cancellation(self):
        z = torch.randn(32, 1, dtype=torch.float64)
        angles, forward_det = self.model.to_angles(z, 13.0)
        recovered, inverse_det = self.model.to_base(angles, 13.0)
        torch.testing.assert_close(recovered, z, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(forward_det, -inverse_det, atol=1e-8, rtol=1e-8)

    def test_reported_jacobian_matches_autograd(self):
        z = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
        angle, log_det = self.model.to_angles(z, 15.0)
        derivative = torch.autograd.grad(angle.sum(), z)[0]
        torch.testing.assert_close(torch.log(derivative.abs()).reshape(-1), log_det)

    def test_support_and_density_normalization(self):
        angles = torch.tensor([[-1], [0], [90], [95], [35]], dtype=torch.float64)
        log_density = self.model.log_prob(angles, 13.0)
        self.assertTrue(torch.isneginf(log_density[:4]).all())
        self.assertTrue(torch.isfinite(log_density[-1]))
        draws = self.model.sample(10_000, 13.0)
        self.assertEqual(draws.shape, (10_000, 1))
        self.assertTrue(((draws > 0) & (draws < 90)).all())
        grid = (torch.arange(8000, dtype=torch.float64) + 0.5)[:, None] * 90 / 8000
        with torch.no_grad():
            mass = self.model.log_prob(grid, 13.0).exp().sum() * 90 / 8000
        self.assertAlmostEqual(mass.item(), 1.0, places=4)

    def test_finite_training_gradients(self):
        angle = torch.rand(64, 1, dtype=torch.float64) * 89 + 0.5
        distance = torch.rand(64, 1, dtype=torch.float64) * 15
        loss = -self.model.log_prob(angle, distance).mean()
        loss.backward()
        gradients = [p.grad for p in self.model.parameters()]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_saved_standardization_and_density(self):
        distances = torch.tensor([[10], [12], [14], [16]], dtype=torch.float64)
        self.model.fit_standardization(distances)
        torch.testing.assert_close(self.model.distance_mean, distances.mean(0))
        torch.testing.assert_close(self.model.distance_std, distances.std(0, unbiased=False))
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        reloaded = ConditionalAngleFlow(hidden_size=8).double()
        reloaded.load_state_dict(torch.load(buffer, weights_only=True))
        angles = torch.tensor([[24.0], [66.0]], dtype=torch.float64)
        torch.testing.assert_close(
            reloaded.log_prob(angles, 13.0), self.model.log_prob(angles, 13.0)
        )


if __name__ == "__main__":
    unittest.main()

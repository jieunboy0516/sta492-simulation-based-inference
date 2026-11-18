"""Numerical checks for the mathematics; run with python3 -m unittest -v."""

import io
import unittest

import torch

from flow import AffineCoupling, ConditionalFlow


class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.model = ConditionalFlow(observation_size=3, hidden_size=8).double()
        # Test nontrivial learned transformations, not just identity initialization.
        for layer in self.model.layers:
            torch.nn.init.normal_(layer.network[-1].weight, std=0.08)
            torch.nn.init.normal_(layer.network[-1].bias, std=0.08)
        self.observation = torch.tensor([0.1, -0.4, 0.8], dtype=torch.float64)

    def test_coupling_inverse_for_both_coordinate_choices(self):
        for kept_coordinate in (0, 1):
            layer = AffineCoupling(3, 8, kept_coordinate).double()
            torch.nn.init.normal_(layer.network[-1].weight, std=0.1)
            values = torch.randn(12, 2, dtype=torch.float64)
            context = self.observation.expand(12, -1)
            transformed, forward_log_det = layer(values, context)
            recovered, inverse_log_det = layer.inverse(transformed, context)
            torch.testing.assert_close(recovered, values, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(forward_log_det, -inverse_log_det)

    def test_entire_flow_inverse_and_determinant_cancellation(self):
        z = torch.randn(32, 2, dtype=torch.float64)
        theta, forward_log_det = self.model.to_parameters(z, self.observation)
        recovered_z, inverse_log_det = self.model.to_base(theta, self.observation)
        torch.testing.assert_close(recovered_z, z, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(forward_log_det, -inverse_log_det)

    def test_reported_jacobian_matches_autograd(self):
        z = torch.tensor([0.4, -0.7], dtype=torch.float64, requires_grad=True)

        def transformation(point):
            theta, _ = self.model.to_parameters(point[None, :], self.observation)
            return theta[0]

        jacobian = torch.autograd.functional.jacobian(transformation, z)
        _, actual_log_det = torch.linalg.slogdet(jacobian)
        _, reported_log_det = self.model.to_parameters(z[None, :], self.observation)
        torch.testing.assert_close(actual_log_det, reported_log_det[0])

    def test_support_and_near_boundary_densities(self):
        theta = torch.tensor(
            [[-1.1, 0], [1, 0], [0, -2], [0, 2.1], [-0.999999, 1.999999]],
            dtype=torch.float64,
        )
        values = self.model.log_prob(theta, self.observation)
        self.assertTrue(torch.isneginf(values[:4]).all())
        self.assertTrue(torch.isfinite(values[4]))
        draws = self.model.sample(10000, self.observation)
        self.assertEqual(draws.shape, (10000, 2))
        self.assertTrue(((draws > self.model.lower) & (draws < self.model.upper)).all())

    def test_training_gradients_are_finite_and_nonzero(self):
        theta = torch.rand(64, 2, dtype=torch.float64) * torch.tensor([2, 4])
        theta = theta + self.model.lower
        observations = torch.randn(64, 3, dtype=torch.float64)
        loss = -self.model.log_prob(theta, observations).mean()
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters()]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_identity_flow_density_integrates_to_one(self):
        # An independent quadrature checks the density and sigmoid correction.
        model = ConditionalFlow(observation_size=3, num_layers=0).double()
        size = 300
        a = -1 + (torch.arange(size, dtype=torch.float64) + 0.5) * 2 / size
        b = -2 + (torch.arange(size, dtype=torch.float64) + 0.5) * 4 / size
        aa, bb = torch.meshgrid(a, b, indexing="ij")
        theta = torch.stack([aa.flatten(), bb.flatten()], dim=1)
        mass = torch.exp(model.log_prob(theta, self.observation)).sum() * 8 / size**2
        self.assertAlmostEqual(mass.item(), 1.0, places=4)

    def test_saved_weights_include_training_standardization(self):
        training_data = torch.randn(200, 3, dtype=torch.float64) * 2 + 4
        self.model.fit_observation_standardization(training_data)
        torch.testing.assert_close(self.model.observation_mean, training_data.mean(0))
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        reloaded = ConditionalFlow(observation_size=3, hidden_size=8).double()
        reloaded.load_state_dict(torch.load(stream, weights_only=True))
        theta = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
        torch.testing.assert_close(
            reloaded.log_prob(theta, self.observation),
            self.model.log_prob(theta, self.observation),
        )


if __name__ == "__main__":
    unittest.main()

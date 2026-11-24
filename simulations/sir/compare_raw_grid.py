"""Reproduce the raw-map comparison in the SIR report."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from sir import OBSERVED_SEED, OBSERVED_THETA, simulate_visits, summarize

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
sys.path.insert(0, str(PROJECT.parent / "line-regressor"))
from flow import ConditionalFlow


class RawGridNPE(nn.Module):
    """Turn the full maps into 16 learned features, then estimate the posterior."""

    def __init__(self):
        super().__init__()
        # The three input channels are the three visits. Each convolution
        # looks at small patches of cells, keeping track of their positions.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(12, 20, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(20, 24, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(24 * 3 * 3, 48),
            nn.SiLU(),
            nn.Linear(48, 16),
        )
        self.flow = ConditionalFlow(16, lower=(0.0, 0.0), upper=(1.0, 1.0))

    def log_prob(self, theta, maps):
        # maps: (batch, 3, 24, 24). features: (batch, 16).
        features = self.encoder(maps.float())
        return self.flow.log_prob(theta, features)


def replay_outbreaks(dataset_sizes):
    """Recreate the notebook's simulated outbreaks in the same order."""
    rng = np.random.default_rng(302)
    parameters = rng.random((sum(dataset_sizes.values()), 2))
    masks = [
        simulate_visits(parameters[start : start + 512], rng)
        for start in range(0, len(parameters), 512)
    ]
    maps = np.concatenate(masks)
    # We only compute summaries for the separate summary-model baselines.
    summaries = np.log1p(summarize(maps)).astype(np.float32)
    theta = torch.tensor(parameters, dtype=torch.float32)
    return theta, torch.from_numpy(maps), torch.from_numpy(summaries)


@torch.no_grad()
def score_outbreaks(model, theta, observations, indices):
    """Score each held-out outbreak at the parameters used to simulate it."""
    model.eval()
    return torch.cat([
        model.log_prob(theta[batch], observations[batch])
        for batch in indices.split(512)
    ])


def train_model(model, theta, observations, n_train, validation_indices,
                batch_size, max_epochs, patience):
    """Train the model, then restore the best validation weights."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_score = -float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in torch.randperm(n_train).split(batch_size):
            loss = -model.log_prob(theta[batch], observations[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_score = score_outbreaks(model, theta, observations, validation_indices).mean().item()
        if not np.isfinite(validation_score):
            raise RuntimeError("The validation score became non-finite.")
        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch}: validation log density {validation_score:.3f}", flush=True)
        if epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_weights)
    return best_epoch, epoch


def saved_summary_scores(saved, theta, summaries, test_indices):
    """Use the two flows already fitted by the notebook in the full comparison."""
    scores = {}
    parameter_counts = {}
    for name, width in (("counts", 3), ("counts_overlap", 5)):
        flow = ConditionalFlow(width, lower=(0.0, 0.0), upper=(1.0, 1.0))
        flow.load_state_dict(torch.load(RESULTS / f"{name}_flow.pt", weights_only=True))
        flow.eval()
        with torch.no_grad():
            scores[name] = flow.log_prob(theta[test_indices], summaries[test_indices, :width])
        np.testing.assert_allclose(
            float(scores[name].mean()), saved["models"][name]["mean_test_log_density"],
            rtol=0, atol=5e-5,
        )
        parameter_counts[name] = sum(weight.numel() for weight in flow.parameters())
    return scores, parameter_counts


def benchmark_direct(theta, maps, n_train, validation_indices, test_indices,
                     batch_size, max_epochs, patience):
    """Condition the flow on the 1,728 original binary values."""
    raw_inputs = maps.flatten(start_dim=1).float()
    torch.manual_seed(407)
    flow = ConditionalFlow(raw_inputs.shape[1], lower=(0.0, 0.0), upper=(1.0, 1.0))

    print(f"Training direct NPE on {n_train:,} outbreaks.", flush=True)
    best_epoch, epochs_run = train_model(
        flow, theta, raw_inputs, n_train, validation_indices,
        batch_size, max_epochs, patience,
    )
    scores = score_outbreaks(flow, theta, raw_inputs, test_indices)
    if not scores.isfinite().all():
        raise RuntimeError("Direct NPE produced a non-finite test score.")

    observed = simulate_visits(OBSERVED_THETA, np.random.default_rng(OBSERVED_SEED))
    observed_input = torch.from_numpy(observed).flatten(start_dim=1).float()
    torch.manual_seed(601)
    draws = flow.sample(8000, observed_input)
    if not ((draws >= 0) & (draws <= 1)).all():
        raise RuntimeError("Direct NPE produced samples outside [0, 1].")

    return {
        "train_outbreaks": n_train,
        "validation_outbreaks": len(validation_indices),
        "test_outbreaks": len(test_indices),
        "observation_size": flow.observation_size,
        "best_validation_epoch": best_epoch,
        "epochs_run": epochs_run,
        "trainable_weights": sum(weight.numel() for weight in flow.parameters()),
        "mean_test_log_density": float(scores.mean()),
        "posterior_mean": draws.mean(dim=0).tolist(),
        "posterior_marginal_90_interval": np.quantile(draws.numpy(), [0.05, 0.95], axis=0).T.tolist(),
    }


def save_results(metrics, filename):
    output = RESULTS / filename
    output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {output.relative_to(PROJECT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-outbreaks", type=int, choices=[2000, 36000], default=36000)
    parser.add_argument("--model", choices=["cnn", "direct"], default="cnn",
                        help="Use a CNN encoder or pass the flattened maps to the flow.")
    args = parser.parse_args()
    n_train = args.training_outbreaks
    torch.set_num_threads(2)
    saved = json.loads((RESULTS / "metrics.json").read_text())
    sizes = saved["dataset_sizes"]
    original_train_end = sizes["train"]
    test_start = original_train_end + sizes["validation"]
    theta, maps, summaries = replay_outbreaks(sizes)
    test_indices = torch.arange(test_start, len(theta))

    # Use the same early stopping and batch sizes as the runs in the report.
    if n_train == 36000:
        validation_indices = torch.arange(original_train_end, test_start)
        batch_size, max_epochs, patience = 512, 70, 10
    else:
        validation_indices = torch.arange(original_train_end, original_train_end + 1000)
        batch_size, max_epochs, patience = 128, 50, 8

    if args.model == "direct":
        metrics = benchmark_direct(
            theta, maps, n_train, validation_indices, test_indices,
            batch_size, max_epochs, patience,
        )
        save_results(metrics, f"direct_comparison_{n_train}.json")
        return

    if n_train == 36000:
        scores, parameter_counts = saved_summary_scores(saved, theta, summaries, test_indices)
        filename = "raw_comparison.json"
    else:
        scores = {}
        filename = "small_budget_comparison.json"
        print("Training the counts-plus-overlaps model on 2,000 outbreaks.", flush=True)
        torch.manual_seed(407)
        summary_flow = ConditionalFlow(5, lower=(0.0, 0.0), upper=(1.0, 1.0))
        summary_flow.fit_observation_standardization(summaries[:n_train])
        train_model(summary_flow, theta, summaries, n_train, validation_indices,
                    batch_size, max_epochs, patience)
        scores["counts_overlap"] = score_outbreaks(summary_flow, theta, summaries, test_indices)

    print(f"Training the raw-map model on {n_train:,} outbreaks.", flush=True)
    torch.manual_seed(407)
    raw_model = RawGridNPE()
    best_epoch, epochs_run = train_model(raw_model, theta, maps, n_train, validation_indices,
                                         batch_size, max_epochs, patience)
    scores["raw_grid"] = score_outbreaks(raw_model, theta, maps, test_indices)

    # A positive difference favors the summary model on the same test cases.
    score_gap = scores["counts_overlap"] - scores["raw_grid"]
    metrics = {
        "test_outbreaks": len(test_indices),
        "mean_test_log_density": {name: float(score.mean()) for name, score in scores.items()},
        "summary_vs_raw_log_density_gain": float(score_gap.mean()),
        "summary_vs_raw_paired_standard_error": float(score_gap.std()) / np.sqrt(len(score_gap)),
    }
    if n_train == 36000:
        parameter_counts["raw_grid"] = sum(weight.numel() for weight in raw_model.parameters())
        observed = simulate_visits(OBSERVED_THETA, np.random.default_rng(OBSERVED_SEED))
        with torch.no_grad():
            torch.manual_seed(601)
            features = raw_model.encoder(torch.from_numpy(observed).float())
            draws = raw_model.flow.sample(8000, features).numpy()
        metrics.update(
            raw_best_validation_epoch=best_epoch,
            raw_epochs_run=epochs_run,
            total_trainable_parameters=parameter_counts,
            posterior_means={
                "counts": saved["models"]["counts"]["posterior_mean"],
                "counts_overlap": saved["models"]["counts_overlap"]["posterior_mean"],
                "raw_grid": draws.mean(axis=0).tolist(),
            },
            raw_observed_marginal_90_intervals=np.quantile(draws, [0.05, 0.95], axis=0).T.tolist(),
        )
    else:
        metrics["train_outbreaks"] = n_train

    save_results(metrics, filename)


if __name__ == "__main__":
    main()

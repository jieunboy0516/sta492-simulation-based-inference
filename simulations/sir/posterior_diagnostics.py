"""Compare held-out simulator pairs with pairs made from posterior draws."""

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

from compare_raw_grid import RESULTS, TRAINING_BUDGETS, replay_outbreaks
from plot_comparison import MODEL_STYLES, read_results
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


def c2st_features(maps, summaries):
    """Give the classifier identical observation features for both classes."""
    cases, visits, _, _ = maps.shape
    # Keep coarse spatial information by counting infectious cells in
    # each non-overlapping 4 x 4 block of each map.
    block_counts = maps.reshape(cases, visits, 6, 4, 6, 4).sum(dim=(3, 5))
    return np.column_stack([summaries.numpy(), block_counts.reshape(cases, -1).numpy()])


def classifier_accuracy(features, theta, posterior, train, test):
    """Train a binary classifier and score it on disjoint outbreak pairs."""
    real = np.column_stack([features, theta])
    simulated = np.column_stack([features, posterior])
    classifier = HistGradientBoostingClassifier(max_iter=140, min_samples_leaf=24,
                                                early_stopping=False, random_state=987)
    classifier.fit(np.vstack([real[train], simulated[train]]),
                   np.repeat([0, 1], len(train)))
    # Treat the two rows for a given outbreak as a pair for the standard error.
    real_correct = classifier.predict(real[test]) == 0
    generated_correct = classifier.predict(simulated[test]) == 1
    pair_accuracy = (real_correct.astype(float) + generated_correct) / 2
    return float(pair_accuracy.mean()), float(pair_accuracy.std(ddof=1) / np.sqrt(len(test)))


def validation_bandwidths(maps, theta):
    """Choose fixed bandwidths from the validation outbreaks."""
    rng = np.random.default_rng(503)
    a, b = rng.integers(len(maps), size=(2, 20_000))
    map_d2 = ((maps[a] != maps[b]).sum(dim=(1, 2, 3))).numpy()
    parameter_d2 = (theta[a] - theta[b]).square().sum(dim=1).numpy()
    return float(np.median(map_d2[map_d2 > 0])), float(np.median(parameter_d2[parameter_d2 > 0]))


def gaussian_kernel(a, b, bandwidth_squared):
    """Gaussian similarities between every row of a and every row of b."""
    d2 = a.square().sum(dim=1)[:, None] + b.square().sum(dim=1)[None, :] - 2 * (a @ b.T)
    return torch.exp(-d2.clamp_min(0) / (2 * bandwidth_squared))


def mmd_squared(maps, theta, candidates, bandwidths):
    """Estimate squared joint MMD with the Gaussian product kernel.

    Each candidate is paired with the true parameter for the same map.
    Exclude all terms comparing an outbreak with itself.
    """
    map_scale, parameter_scale = bandwidths
    flat = maps.flatten(start_dim=1).float()
    totals = dict.fromkeys(candidates, 0.0)
    for start in range(0, len(theta), 256):
        stop = min(start + 256, len(theta))
        batch = slice(start, stop)
        map_kernel = gaussian_kernel(flat[batch], flat, map_scale)
        true_kernel = gaussian_kernel(theta[batch], theta, parameter_scale)
        for name, draws in candidates.items():
            model_kernel = gaussian_kernel(draws[batch], draws, parameter_scale)
            true_to_model = gaussian_kernel(theta[batch], draws, parameter_scale)
            model_to_true = gaussian_kernel(draws[batch], theta, parameter_scale)
            terms = map_kernel * (true_kernel + model_kernel - true_to_model - model_to_true)
            terms[torch.arange(stop - start), torch.arange(start, stop)] = 0
            totals[name] += terms.sum(dtype=torch.float64).item()
    pairs = len(theta) * (len(theta) - 1)
    return {name: value / pairs for name, value in totals.items()}


def load_candidates(theta, test_indices):
    candidates = {}
    for n_train in TRAINING_BUDGETS:
        for model in ("cnn", "direct"):
            with np.load(RESULTS / f"test_draws_{model}_{n_train}.npz") as archive:
                np.testing.assert_array_equal(archive["theta"], theta.numpy())
                np.testing.assert_array_equal(archive["test_indices"], test_indices.numpy())
                for name in archive.files:
                    if name in MODEL_STYLES:
                        draws = archive[name]
                        if draws.shape != tuple(theta.shape) or not ((0 <= draws) & (draws <= 1)).all():
                            raise ValueError(f"Invalid posterior draws for {name} at {n_train:,} outbreaks.")
                        candidates[n_train, name] = torch.from_numpy(draws)
    expected = {
        (budget, model)
        for budget in TRAINING_BUDGETS
        for model in MODEL_STYLES
    }
    if set(candidates) != expected:
        raise ValueError("The saved draws are missing a model or training budget.")
    return candidates


def plot_c2st(results):
    fig, ax = plt.subplots(figsize=(7.6, 2.45), layout="constrained")
    ax.axhline(0.5, color="#596164", linestyle="--", lw=1.1, zorder=1)
    markers = ("o", "s", "^", "D")
    bounds = [0.5]
    handles = []
    for (name, (label, color)), marker in zip(MODEL_STYLES.items(), markers):
        points = [results[str(n)][name] for n in TRAINING_BUDGETS]
        accuracy = np.array([point["c2st_accuracy"] for point in points])
        errors = np.array([point["c2st_se"] for point in points])
        bounds.extend(accuracy - errors)
        bounds.extend(accuracy + errors)
        ax.errorbar(TRAINING_BUDGETS, accuracy, yerr=errors, color=color,
                    fmt=f"{marker}-", lw=1.5, capsize=2.8, markersize=4.8, zorder=2)
        handles.append(Line2D([], [], marker=marker, color=color, label=label, lw=1.5))

    ax.set_ylim(min(bounds) - 0.014, max(bounds) + 0.029)
    ax.set_xlim(0, 39_000)
    ax.set_xticks(TRAINING_BUDGETS, [f"{n:,}" for n in TRAINING_BUDGETS])
    ax.set(xlabel="Training outbreaks", ylabel="C2ST accuracy")
    # Make the chance reference an explicit tick on the percentage axis.
    y_min, y_max = ax.get_ylim()
    steps = range(int(np.ceil(y_min * 20)), int(np.floor(y_max * 20)) + 1)
    ax.set_yticks([step / 20 for step in steps])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dfe4e4", lw=0.7)
    ax.legend(handles=handles, frameon=False, ncols=4, handletextpad=0.45, columnspacing=1.0,
              loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=9)
    fig.savefig(RESULTS / "c2st_comparison.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_mmd(results):
    fig, ax = plt.subplots(figsize=(7.6, 2.45), layout="constrained")
    ax.axhline(0.0, color="#596164", linestyle="--", lw=1.1)
    extrema = [0.0]
    for center, group in enumerate(results.values()):
        order = [name for name in MODEL_STYLES if name in group]
        for i, name in enumerate(order):
            score = 1000 * group[name]["mmd_squared"]
            extrema.append(score)
            x = center + (i - (len(order) - 1) / 2) * 0.20
            color = MODEL_STYLES[name][1]
            ax.errorbar(x, score, color=color, fmt="o", capsize=3, markersize=5)
            ax.annotate(f"{score:.3f}", (x, score), xytext=(0, 4),
                        textcoords="offset points", ha="center", color=color, fontsize=9)

    low, high = min(extrema), max(extrema)
    pad = max(high - low, 0.01)
    ax.set_ylim(low - 0.14 * pad, high + 0.24 * pad)
    ax.set_xlim(-0.44, 1.44)
    ax.set_xticks([0, 1], ["2,000 training outbreaks", "36,000 training outbreaks"])
    ax.set_ylabel(r"Squared MMD $\times 10^3$")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dfe4e4", lw=0.7)
    handles = [Line2D([], [], marker="o", linestyle="none", color=color, label=label)
               for label, color in MODEL_STYLES.values()]
    ax.legend(handles=handles, frameon=False, ncols=4, handletextpad=0.3, columnspacing=1.0,
              loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0, fontsize=9)
    fig.savefig(RESULTS / "mmd_comparison.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    sizes = read_results("metrics.json")["dataset_sizes"]
    theta, maps, summaries = replay_outbreaks(sizes)
    validation = slice(sizes["train"], sizes["train"] + sizes["validation"])
    bandwidths = validation_bandwidths(maps[validation], theta[validation])
    test = torch.arange(validation.stop, len(theta))
    theta, maps, summaries = theta[test], maps[test], summaries[test]
    candidates = load_candidates(theta, test)

    # Split by outbreak so both rows for that outbreak stay on the same side.
    classifier_train, classifier_test = np.split(np.random.default_rng(882).permutation(len(test)), 2)
    features = c2st_features(maps, summaries)
    distances = mmd_squared(maps, theta, candidates, bandwidths)
    results = {}
    for (n_train, name), draws in candidates.items():
        accuracy, se = classifier_accuracy(features, theta.numpy(), draws.numpy(),
                                           classifier_train, classifier_test)
        results.setdefault(str(n_train), {})[name] = {
            "c2st_accuracy": accuracy,
            "c2st_se": se,
            "mmd_squared": distances[n_train, name],
        }
        print(f"{n_train:>5} {name:>15}: C2ST {accuracy:.2%}, MMD² {distances[n_train, name]:.6f}",
              flush=True)

    output = {
        "test_outbreaks": len(test),
        "c2st_classifier_training_outbreaks": len(classifier_train),
        "c2st_classifier_test_outbreaks": len(classifier_test),
        "map_bandwidth_squared": bandwidths[0],
        "parameter_bandwidth_squared": bandwidths[1],
        "budgets": results,
    }
    (RESULTS / "posterior_diagnostics.json").write_text(json.dumps(output, indent=2) + "\n")
    plot_c2st(results)
    plot_mmd({n: results[n] for n in ("2000", "36000")})


if __name__ == "__main__":
    main()

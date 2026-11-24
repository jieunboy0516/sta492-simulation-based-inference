"""Regenerate the SIR report's benchmark figures from the saved results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
MODEL_STYLES = {
    "counts": ("Counts only", "#75818e"),
    "counts_overlap": ("Counts + overlaps", "#176c61"),
    "raw_grid": ("Raw maps + CNN", "#395a89"),
    "direct": ("Raw maps + direct NPE", "#ad5d26"),
}


def read_results(filename):
    return json.loads((RESULTS / filename).read_text())


def plot_test_scores(small, full, direct_small, direct_full):
    """Compare mean log densities at the true parameters on the same test set."""
    fig, ax = plt.subplots(figsize=(7.7, 3.5), layout="constrained")
    all_scores = []
    for center, (baseline, direct) in enumerate(((small, direct_small), (full, direct_full))):
        scores = baseline["mean_test_log_density"] | {"direct": direct["mean_test_log_density"]}
        group = [model for model in MODEL_STYLES if model in scores]
        for i, model in enumerate(group):
            score = scores[model]
            all_scores.append(score)
            x = center + (i - (len(group) - 1) / 2) * 0.176
            bar = ax.bar(x, score, width=0.151, color=MODEL_STYLES[model][1])
            ax.bar_label(bar, labels=[f"{score:.2f}"], padding=3, fontsize=10)

    ax.set_ylabel("Mean test log density (nats)", fontsize=11)
    ax.set_xticks([0, 1], ["2,000 training outbreaks", "36,000 training outbreaks"], fontsize=11)
    ax.set_ylim(min(0, min(all_scores) - 0.3), max(all_scores) + 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dfe4e4", lw=0.7)
    handles = [Patch(facecolor=color, label=label) for label, color in MODEL_STYLES.values()]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
              borderaxespad=0, ncols=4, columnspacing=1.2, handlelength=1.35, fontsize=9.5)
    fig.savefig(RESULTS / "summary_comparison.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_presentation_fit(summary, full, direct):
    """Compare model sizes and the intervals for the one presentation outbreak."""
    models = ["counts_overlap", "raw_grid", "direct"]
    colors = [MODEL_STYLES[model][1] for model in models]
    positions = [2, 1, 0]
    weights = [full["total_trainable_parameters"][model] for model in models[:2]]
    weights.append(direct["trainable_weights"])
    means = np.array([
        full["posterior_means"]["counts_overlap"], full["posterior_means"]["raw_grid"],
        direct["posterior_mean"],
    ])
    intervals = np.array([
        summary["models"]["counts_overlap"]["posterior_marginal_90_interval"],
        full["raw_observed_marginal_90_intervals"],
        direct["posterior_marginal_90_interval"],
    ])
    truth = summary["observed_parameters"]

    fig, axes = plt.subplots(1, 3, figsize=(7.7, 2.55), sharey=True,
                             gridspec_kw={"width_ratios": [1.1, 1, 1]}, layout="constrained")
    for ax in axes:
        ax.set_ylim(-0.4, 2.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9, length=3)
        ax.tick_params(axis="y", length=0, labelsize=9.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dfe4e4", lw=0.7)

    size_ax = axes[0]
    size_ax.set_title("Trainable weights\n(log scale)", fontsize=10.5)
    size_ax.set_xscale("log")
    size_ax.set_xlim(20_000, 2_300_000)
    size_ax.set_xticks([30_000, 100_000, 300_000, 1_000_000], ["30k", "100k", "300k", "1M"])
    size_ax.minorticks_off()
    size_ax.set_yticks(positions, ["Counts + overlaps", "Raw maps + CNN", "Raw maps + direct NPE"])
    for y, weight, color in zip(positions, weights, colors):
        size_ax.scatter(weight, y, color=color, s=27)
        size_ax.annotate(f"{weight:,}", (weight, y), xytext=(6, 0), textcoords="offset points",
                         va="center", fontsize=9)

    for i, (ax, label) in enumerate(zip(axes[1:], ["Transmission $p$", "Recovery $q$"])):
        values = [truth[i], *means[:, i], *intervals[:, i, :].ravel()]
        low, high = min(values), max(values)
        pad = max((high - low) * 0.12, 0.005)
        ax.set_xlim(max(0, low - pad), min(1, high + pad))
        ax.set_title(f"{label}\nTrue value {truth[i]:.2f}", fontsize=10.5)
        ax.axvline(truth[i], color="#30343a", linestyle=(0, (3, 2)), lw=1.1)
        ax.tick_params(axis="y", labelleft=False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f" if i == 0 else "%.3f"))
        for row, (y, color) in enumerate(zip(positions, colors)):
            mean = means[row, i]
            start, end = intervals[row, i]
            ax.hlines(y, start, end, color=color, lw=1.35)
            ax.scatter(mean, y, color=color, s=27)
            ax.annotate(f"{mean:.3f}", (mean, y), xytext=(6, 7),
                        textcoords="offset points", ha="left", fontsize=9, color=color)

    fig.savefig(RESULTS / "summary_tradeoff.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main():
    small = read_results("small_budget_comparison.json")
    full = read_results("raw_comparison.json")
    direct_small = read_results("direct_comparison_2000.json")
    direct_full = read_results("direct_comparison_36000.json")
    if {run["test_outbreaks"] for run in (small, full, direct_small, direct_full)} != {4000}:
        raise ValueError("The comparison expects all models to use the 4,000 test outbreaks.")
    plot_test_scores(small, full, direct_small, direct_full)
    plot_presentation_fit(read_results("metrics.json"), full, direct_full)


if __name__ == "__main__":
    main()

# Ball projectile: which angle explains a landing distance?

Start with **[projectile_demo.ipynb](projectile_demo.ipynb)**. The saved notebook
contains explanations, an executed training run, plots, and numerical checks.
The worked mathematical explanation is **[report.pdf](report.pdf)**, with editable
source in **[report.tex](report.tex)**.

We observe a landing distance and infer a distribution over the launch angle.
A low launch and a high launch can land at the same distance, so the posterior
can have two peaks. One trained normalizing flow handles observed distances of
13, 15, and 16 metres without retraining.

## The model, in plain language

The ball leaves the ground at 13 m/s. Gravity is 9.81 m/s². Each throw has a
new random tailwind acceleration and new distance-measurement noise. We infer
only the angle; the random wind is part of the simulator's uncertainty.

- The angle prior is Normal with mean 35 degrees and standard deviation 15
  degrees, truncated to the interval from 0 to 90 degrees.
- Wind is a constant horizontal acceleration during a throw, drawn from an
  Exponential distribution with rate 2 and mean 0.5 m/s². It is not a wind speed.
- Measurement noise is Normal with standard deviation 0.25 metres.
- We omit air resistance. We calculate the trajectory and landing distance
  directly, with no numerical time stepping.

This is a simplified adaptation of Section 2 and Figure 2 of *Simulation-Based
Inference: A Practical Guide*. The guide's code includes air resistance and
different numerical settings. The adaptation keeps the inverse problem and
stochastic structure while making the equations easier to derive.

All public angle inputs, samples, and densities use **degrees**. The physics
functions convert to radians for trigonometry. Only the final landing distance
is used for inference; trajectories are illustrations. The three observations
are hypothetical distances, not datasets with one known generating angle.

## Read the files in this order

1. The notebook explains the forward model, prior, NPE objective, and flow.
2. `projectile.py`: `trajectory` shows the path, and `simulate` draws random
   measured ranges. The reference likelihood/posterior functions at the end
   are for validation only.
3. `flow.py`: a small network predicts the spline shape from a distance;
   `to_angles`, `to_base`, and `log_prob` track the change in density. Only the
   spline arithmetic comes from `nflows`.
4. Return to the notebook for the visible training loop and measured comparisons.
5. The report gives the derivations, including the spline formula and derivative.

## Set up and run

Use Python 3.10 or later. From this folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m jupyterlab projectile_demo.ipynb
```

Choose **Restart Kernel and Run All**, or run the notebook from a fresh kernel
and save its outputs with:

```sh
python3 run_notebook.py
```

The default run uses the CPU and two PyTorch threads. It generates 20,000 pairs,
reserves 2,000 for validation, and trains a three-layer conditional spline flow.
Actual runtime, seeds, versions, and training settings are in `results/metrics.json`.
Re-running replaces the saved experiment results. One training seed is used;
the experiment is not a repeated-dataset coverage or calibration study.

For a quick learning run, reduce `NUM_SIMULATIONS` and `MAX_EPOCHS` in the
notebook. This can reduce posterior accuracy. Changes to the simulator or prior
require retraining. A different observed distance under the same setup does not.

## Use the trained flow

Run from this folder after installing the dependencies:

```python
import torch
from flow import ConditionalAngleFlow

checkpoint = torch.load("results/angle_flow.pt", map_location="cpu", weights_only=True)
model = ConditionalAngleFlow(**checkpoint["model_kwargs"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

angles = model.sample(1000, distance_m=13.0)  # shape (1000, 1), degrees
log_density = model.log_prob(angles[:5], 13.0)  # shape (5,), density is per degree
```

Distance standardization is saved in the model weights. The public interface
accepts angle tensors of shape `(B, 1)` and distance tensors of shape `(B, 1)`
or one scalar shared by the batch. Sampling accepts one observed distance.
The flow enforces the angle bounds with an invertible sigmoid transformation,
not by clipping samples.

## Results and validation

`results/` contains the model weights, three sets of posterior draws and density
grids, metrics, loss history, generated report tables, and PNG/PDF figures.

We can evaluate a numerical reference for this toy model by integrating out wind
and applying Bayes' rule over angle. That calculation is separate from NPE
training. We compare 2,000- and 4,000-point grids covering the entire angle prior.
The learned density is **never normalized again** on the grid. Reported metrics
include its original integral, total variation distance from the reference, and
posterior probability below 45 degrees.

Read both peaks and their probability weights. A mean angle can lie between
the peaks, and a central credible interval can include a low-density gap.
Posterior predictive ranges use new wind and new measurement noise.

Run the ten mathematical and numerical checks with:

```sh
python3 -m unittest -v
```

They check landing equations, wind effects, simulated moments, prior support,
reference-grid refinement, flow inversion, automatic-differentiation Jacobians,
density normalization, finite gradients, and saved standardization.

## Build the mathematical report

With `pdflatex` installed:

```sh
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

The notebook generates the numerical table and training macros used by the
report. After a changed run, rebuild the PDF and review its interpretation too.

## References

- Deistler et al. (2025), [Simulation-Based Inference: A Practical Guide](https://arxiv.org/abs/2508.12939), Section 2 and Figure 2.
- [The authors' ball-throw implementation](https://github.com/sbi-dev/sbi-practical-guide/blob/main/examples/ball_throw.ipynb).
- Weniger (2026), [Lecture 1: ABC and Neural SBI](https://cweniger.github.io/teaching-2608-IAIFI/lecture1.html).
- Durkan et al. (2019), [Neural Spline Flows](https://arxiv.org/abs/1906.04032).
- [nflows: normalizing flows in PyTorch](https://github.com/bayesiains/nflows), version 0.14.

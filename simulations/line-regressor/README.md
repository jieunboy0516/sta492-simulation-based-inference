# NPE with a small normalizing flow

Start with **[npe_demo.ipynb](npe_demo.ipynb)**. It explains the experiment one
step at a time and includes the saved plots and numerical results. Read the
network code in **[flow.py](flow.py)** when you reach the coupling-layer section.
The short write-up is **[report.pdf](report.pdf)**; its editable source is
**[report.tex](report.tex)**.

This is a learning project. SBI means using simulations for inference; NPE
means learning a posterior density from simulated parameter–data pairs; a
normalizing flow is the particular density model used here.

## What the examples do

Both examples infer two parameters, `a` and `b`, from 20 noisy measurements at
known locations `t` between -0.2 and 0.2. Their priors are independent uniforms:
`a` between -1 and 1, and `b` between -2 and 2. Noise has standard deviation 0.25.

- **Linear:** `y = a + b*t + noise`.
- **Curved posterior:** `y = a + b*t + b**2 + noise`. The observable intercept is
  now `a + b**2`, so uncertainty about `b` produces a curved relationship with `a`.

The second model is nonlinear in the unknown parameters; its mean measurements
still form a straight line in `t`. Both are deliberately tractable toy models.
The training procedure uses only simulations. A separately evaluated likelihood
lets us check the answers against numerical reference posteriors.

## Open or run

Python 3.10 or later is required. From this folder, create an optional isolated
environment and install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

To open the notebook interactively:

```sh
python3 -m jupyterlab npe_demo.ipynb
```

Use **Restart Kernel and Run All** to reproduce it. Alternatively, this command
executes it in a fresh kernel and saves all outputs back into the notebook:

```sh
python3 run_notebook.py
```

The default run trains two models on the CPU, using 20,000 pairs each, split
into 18,000 training and 2,000 validation pairs. It uses batches of 256, up to
100 epochs, and early stopping after 15 epochs without validation improvement.
The best validation weights are restored. Actual timing and epochs are recorded
in `results/metrics.json`. A GPU is not needed. Re-running overwrites the saved
experiment outputs. Seeds are fixed; different hardware or library versions can
still produce slightly different neural results.

For a quick learning run, reduce `NUM_SIMULATIONS` and `MAX_EPOCHS` in the notebook.
The posterior may then be less accurate. The report describes the supplied full
run; re-running the notebook does not automatically rewrite its prose or tables.

## How to read the code

1. In the notebook, read the simulator, the NPE loss, and the one-layer example.
2. In `flow.py`, read `AffineCoupling.forward` and `inverse`. One coordinate stays
   fixed while the other is scaled and shifted using the data.
3. Read `ConditionalFlow.to_parameters` and `to_base`. These combine the layers
   and the final sigmoid transformation that enforces the prior bounds.
4. Read `log_prob`: base Gaussian density plus the inverse Jacobian correction.
5. Return to the notebook and read `train_flow`, then the posterior diagnostics.

The only model inputs are the parameters and observed measurements. The fixed
locations and noise standard deviation define the simulator. Each flow is trained
for one value of `c`; the value of `c` is not inferred.

## Reuse saved weights

Run this from this folder, or use the final notebook cell:

```python
import torch
from flow import ConditionalFlow

checkpoint = torch.load("results/curved_weights.pt", map_location="cpu", weights_only=True)
model = ConditionalFlow(**checkpoint["model_kwargs"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Replace this with your 20 measurements under the same simulator and prior.
import numpy as np
saved = np.load("results/curved_posterior.npz")
observation = torch.tensor(saved["observation"], dtype=torch.float32)
samples = model.sample(1000, observation)  # shape: (1000, 2)
log_density = model.log_prob(samples[:10], observation)  # shape: (10,)
```

Observation standardization is stored in the model weights. A new dataset from
the same setup needs no retraining. A changed prior, measurement grid, noise
model, or simulator does require retraining.

## Saved results and checks

- `results/figures/`: observations, training curves, joint posteriors, marginals,
  and posterior predictive checks, each as PNG and PDF.
- `results/*_weights.pt`: model weights and initialization settings.
- `results/*_posterior.npz`: posterior draws, observations, grid coordinates,
  reference densities, and learned densities.
- `results/metrics.json`: settings, seeds, versions, timings, posterior summaries,
  grid integrals, total variation distances, and grid refinement comparisons.
- `results/training_history.json`: training and validation loss each epoch.

The reference covers the whole prior rectangle. The linear reference is a
**truncated Gaussian**, because the prior has finite bounds. The learned density
is evaluated with its original normalization; it is never normalized again on
the grid. Grids of 400 and 800 points per axis check numerical accuracy.

Run the mathematical tests with:

```sh
python3 -m unittest -v
```

They check inversion, automatic-differentiation Jacobians, gradients, bounds,
density normalization, and saved observation standardization. The notebook
also checks grid refinement and that loaded weights reproduce density values.
This is one training run and one observed dataset per example, not a calibration
or coverage study across repeated datasets.

To compile the supplied report, install a TeX distribution with `pdflatex` and run:

```sh
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

## References

- Dinh, Sohl-Dickstein and Bengio (2017), [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803).
- Papamakarios and Murray (2016), [Fast epsilon-free inference of simulation models with Bayesian conditional density estimation](https://proceedings.neurips.cc/paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html).

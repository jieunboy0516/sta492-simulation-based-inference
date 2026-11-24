# SIR on a grid with simulation-based inference

This example infers the transmission probability \(p\) and recovery probability
\(q\) from the three infectious-status maps in the presentation. It uses the
same 24 × 24 grid, four starting infections, and observation steps 4, 8, and 12.

Start with the [executed notebook](sir_demo.ipynb). It shows the maps, explains
how three counts and two overlaps are computed, and fits two neural posterior
estimators using the same simulations. Compare their posterior plots and
held-out scores to see what adding overlaps does. The [report](report.pdf)
gives the mathematical derivations and explains the results. The
[LaTeX source](report.tex) is included.

Each summary row is ordered as
\((C_4,C_8,C_{12},O_{4,8},O_{8,12})\). A count records how many cells are
infectious at one visit. An overlap counts how many of those same cells are
infectious at the next visit. The presentation maps give \((33,60,86,17,25)\).

## Rerun locally

From the repository root, with Python 3.10 or newer available:

~~~sh
cd simulations/sir
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_notebook.py
~~~

The runner starts a fresh kernel, retrains both models on CPU, and saves the
executed notebook. To explore it interactively, run `jupyter lab sir_demo.ipynb`.
Fixed seeds make the example reproducible. Small numerical
differences across platforms are possible. Run from this directory so the
notebook can import the existing flow from [line-regressor](../line-regressor/flow.py).

- [sir.py](sir.py) contains the grid update, observation, and summaries.
- [results/metrics.json](results/metrics.json) contains test scores, coverage,
  posterior intervals, and predictive quantiles.
- [results/](results/) contains the figures and both PyTorch state dictionaries.
  The counts model expects 3 inputs and the counts-plus-overlaps model expects
  5 inputs, in the order above. Both expect \(\log(1+s)\), and the fitted
  standardization is already saved in their weights.

## Run the comparison with raw maps

[compare_raw_grid.py](compare_raw_grid.py) trains two kinds of neural posterior
estimators on the three binary maps. The CNN learns 16 features for the flow.
Direct NPE reshapes the three maps into 1,728 binary entries and sends those
entries to the flow. The direct model has 690,060 trainable weights.

Use the same Python environment as the notebook. From this directory, run:

~~~sh
python compare_raw_grid.py --model direct --training-outbreaks 2000
python compare_raw_grid.py --model direct --training-outbreaks 36000
~~~

For the CNN, use \`--model cnn\` or omit \`--model\`. Both comparisons use the
same 4,000 test outbreaks. The CNN commands save the scores for the CNN and
the summary models. The direct commands save their results separately.

| Training outbreaks | Summary and CNN results | Direct NPE results |
| ---: | --- | --- |
| 2,000 | [Scores](results/small_budget_comparison.json) | [Scores and posterior](results/direct_comparison_2000.json) |
| 36,000 | [Scores](results/raw_comparison.json) | [Scores and posterior](results/direct_comparison_36000.json) |

Run \`python plot_comparison.py\` to regenerate the comparison plots from
these saved numbers.

To rebuild the report PDF, run `pdflatex -interaction=nonstopmode -halt-on-error report.tex`
twice from this directory. The report describes the saved run and uses the figures
from `results/`.

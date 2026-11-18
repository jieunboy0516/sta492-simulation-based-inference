# STA492: Simulation-Based Inference

Project repository for exploring the assigned research papers and possible future
research directions in simulation-based inference. The research question remains
a placeholder until it is finalized.

**Group members:** 
- Samsam Lee
- Kian Dianati

## Reference papers

1. Michael Deistler et al. (2025). [Simulation-Based Inference: A Practical Guide](https://arxiv.org/abs/2508.12939). arXiv:2508.12939.
2. Kyle Cranmer, Johann Brehmer, and Gilles Louppe (2019). [The frontier of simulation-based inference](https://arxiv.org/abs/1911.01429). arXiv:1911.01429.

Add or remove paper entries as needed.

## Report workflow

The report outline follows the STA492 report guidelines:

1. Read and critically synthesize the assigned papers and their statistical ideas.
2. Ask AI for five possible research directions based on those papers, then assess
   each for scientific value, statistical soundness, novelty, feasibility, and
   whether it is worth pursuing.
3. Explain which directions are most promising and why. Include ideas developed
   by the group where relevant.
4. Document how AI suggestions were checked and make any computational work
   reproducible, with supporting details in supplementary material when useful.

Write a coherent statistical manuscript, with length guided by substance and
clarity. The course sets no fixed page limit and does not require implementing a
research direction. Give more space to the directions that warrant deeper study.

## Repository layout

```text
.
├── README.md
├── .gitignore
└── report/
    ├── report.tex       # Editable report source
    └── build/           # Generated PDF and LaTeX files; ignored by Git
```

## Build the report on macOS

Use BasicTeX with `pdflatex` and `latexmk` available in your terminal. The starter
uses standard packages already available in the BasicTeX installation used to
check this repository.

Run this command from the repository root:

```sh
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build report/report.tex
```

The PDF is written to **`report/build/report.pdf`**. Re-run the same command after
editing; `latexmk` handles the compilation passes needed for cross-references.

If your terminal cannot find the TeX commands, add BasicTeX's command directory to
the current shell and try again:

```sh
export PATH="/Library/TeX/texbin:$PATH"
```

## Report draft and presentation slides

- [Report draft (Google Docs)](https://docs.google.com/document/d/1-2sz6M0IZ_1TCE7WXlljw7BMHPN3UOZlgvDi-0xLGdE/edit?usp=sharing)
- [Presentation slides (Google Slides)](https://docs.google.com/presentation/d/1UMGA1gGIlv03YM7jHCzXXs9Cm_lsRJMi1Lr9iYceKZc/edit?slide=id.p#slide=id.p)

# Simulation-based Inference — speaker notes

## 1. Simulation-based Inference

Ask the audience to imagine receiving a handful of outbreak maps. Which combination of transmission and recovery could have produced them? We will follow one set of visits through simulation, posterior estimation and model checking.

## 2. Bayesian Recap

Start with a distinction that will matter again at the end. Nature generates observations. We choose the family of distributions through which we interpret them.

## 3. Data Generating Process

Use uppercase $X$ for the random observation and lowercase $x$ for a realized value. At this point $p_0$ stands for the actual distribution of the observation. We have not assumed an epidemic mechanism yet.

## 4. Bayesian Approach

For now suppose the family contains the truth. The prior describes plausible parameter values; the likelihood describes how plausible the observed data are at each value. The denominator makes the posterior integrate to one. For comparing two parameter settings at the same observation it cancels.

## 5. From Bayes to SBI

Advance through the three computational assumptions. In the binomial example the beta prior and likelihood give a beta posterior. In Metropolis–Hastings the acceptance ratio only requires relative target densities, together with the proposal terms. In SBI we can draw parameters and synthetic observations even when evaluating the observed-data likelihood is impractical. Point to the simulator arrow as the final term appears.

## 6. Introduction to Simulation-based Inference

What could make sampling easy and scoring a particular observation hard? A grid with missing time points will give us both situations at once.

## 7. Epidemiology

Ask the two questions separately. Transmission refers to a susceptible person in contact with an infectious neighbor. A newly infected person enters the exposed state for eight steps before becoming infectious. Recovery refers to someone who is already infectious. A snapshot alone can confound a large number of new infections with a long infectious period.

## 8. Let’s make a simulator

Press **Play**, then pause between two visits. The hidden states on the left can change while the latest recorded map on the right stays fixed. A zero on the right can indicate a susceptible, exposed or recovered person. At step 8, no one is visibly infectious, but the hidden grid still contains exposed people. Change the sliders and replay a few steps. Reset restores the original parameter pair and recording. The same random stream makes the settings easy to compare. The grid has four-neighbor contact, irreversible recovery and synchronous updates. With $k$ infectious neighbors the per-step infection probability is $1-(1-\theta_1)^k$.

## 9. Real-world observations

Here “real-world” means the fixed observation we give the estimator in this synthetic experiment. We have eight visits four steps apart. Show the first three here and all eight later. The eight infectious counts are 1, 0, 5, 3, 10, 3, 8 and 4. We know the generating values, approximately $(0.499,0.265)$, because we simulated this example. The estimator receives only the recorded maps. Adjusting the simulator sliders leaves the fixed recording on this slide intact.

## 10. Neural Methods for SBI

We can generate supervised training examples: on each simulation the generating parameter is known. The two methods use the same kind of pairs and reverse what they predict.

## 11. Conditional density estimation — NLE

Point across the diagram. Each superscript identifies a whole simulation, while subscripts on $\theta$ identify its two components. Use a neural network to parameterize the conditional density $q_\phi$. Here $\phi$ denotes the network weights and $\theta$ denotes the simulator parameters. Reveal the loss and minimize it over $\phi$. For each simulated pair, score the density assigned to the simulated observation given its generating parameter. Training minimizes the average negative log density across these pairs. After training, evaluate it at the fixed observation and proposed parameter values. Multiply the learned likelihood by the prior to get an approximate MCMC target.

## 12. Conditional density estimation — NPE

Advance from the crossed-out NLE label to NPE. The approximation and loss reverse direction together. Fit the posterior network’s weights $\phi$ by minimizing $\mathcal L_{\mathrm{NPE}}(\phi)$. Use the same simulated pairs, and now score the density assigned to the generating parameter given its observation. Both expectations average over the joint draw shown at the top of the slide. With training parameters drawn from the prior, the conditional distribution of the generating parameter given a simulated observation is the model posterior. At presentation time, give the fitted estimator the eight observed binary maps and draw parameter pairs directly.

## 13. Transport maps

The left heatmap shows the standard Gaussian base density. The right heatmap estimates the fitted SIR flow’s joint density for transmission and recovery using 16,384 draws conditional on the eight observed binary maps. Each panel scales its color to its own peak. The axes give Gaussian coordinates on the left and probability units on the right. The flow-matching slide animates individual trajectories under this same fit.

## 14. Normalizing flows

Hold $x$ fixed and invert the map to recover the Gaussian input that produced $\theta$. The determinant adjusts the base density for the local change in volume. Figure 1 of [Dax et al. (2023)](https://arxiv.org/pdf/2305.17161#page=2) illustrates a discrete flow for a two-moons example. The observation conditions each layer. Requiring efficient inversion and determinant evaluation restricts the choice of layers.

## 15. Research Direction I: Generative Modeling for SBI

How can we learn a transport map for the posterior by fitting a velocity field?

## 16. Flow matching

Draw a simulation pair $(\theta,x)$, independent Gaussian noise $\eta$, and a uniform time $t$. The straight interpolation between the noise and the parameter has known velocity $\theta-\eta$. Squared-error regression teaches the network the average target velocity conditional on location, time, and observation. To sample for a new observation $x^*$, initialize a fresh particle at $\eta$ and integrate $\dot\theta_t=v_{t,x^*}(\theta_t)$. The training path uses the actual generating parameter pair as its endpoint. A two-dimensional convolutional encoder takes the eight recorded maps as channels. Its output conditions the velocity field. The FMPE panel is the paper’s two-moons example.

<!-- Notes for the omitted score matching slide.
During training, draw $(\theta,x)$ from the simulator joint, a diffusion time $t$ uniformly, and independent standard Gaussian noise $\eta$. The schedule controls how much of the clean parameter remains through $\alpha_t$ and how much noise enters through $\sigma_t$. For a variance-preserving diffusion, $\alpha_t^2+\sigma_t^2=1$.

At a given $t$, the score of the Gaussian perturbation conditional on the clean parameter is $-\eta/\sigma_t$. The slide shows the denoising loss weighted by $\sigma_t^2$. Squared-error regression averages the target over the possible clean parameters given $\theta_t$ and $x$. This conditional average is the score of the noisy posterior. Trace the figure forward to see the two moons disperse, then backward to see how the learned score guides sampling. In this slide’s convention, sampling runs from large $t$ toward zero. The score enters the reverse diffusion drift or the corresponding probability-flow ODE. [Sharrock et al. (2024), Fig. 1 and Eq. (7)](https://proceedings.mlr.press/v235/sharrock24a.html)
-->

## 17. Sampling and density evaluation

Both computations hold the observation fixed. For sampling, start from Gaussian noise and follow the field forward. For density evaluation, start at the query point and trace the trajectory back to its Gaussian starting point. Along the trajectory, positive divergence means a small volume expands and its log density decreases. This gives the minus sign in the integral from zero to one. [Dax et al. (2023), Eqs. (2)–(3)](https://arxiv.org/pdf/2305.17161#page=4)

## 18. Flow matching in the SIR example

Press **Play**. Each orange dot starts as an independent standard Gaussian draw and follows the fitted conditional velocity field in the same two-dimensional space. The neural network was trained on the simulator’s transmission and recovery parameters. With the spatial features of the eight observed maps held fixed, integrate each particle from time zero to one using the midpoint rule. At time one the two components are the estimated transmission and recovery probabilities. Earlier states are intermediate ODE values and can be outside the unit square. White arrows show network evaluations at the current time. The background estimates the current density of 16,384 particles. Both panels have fixed axes throughout playback. The right panel shows the full unit square outlined on the left, with the same particles and relative density scale. The final heatmap is shared with Transport maps.

## 19. Research Direction II: Summary Statistics

Return to the observation side of the same experiment. What could we retain from eight maps if we had room for only a small vector?

## 20. From eight maps to eight counts

Read the row left to right. Each number counts red cells directly above it. We compress $8\times24\times24=4{,}608$ binary measurements to eight numbers. Here we explore what would happen if we compressed the maps. In the ABC example, apply the same transformation to simulated and observed maps, then match on the summaries. It recovers the full-data posterior if the summary is sufficient for these parameters; arbitrary compression gives no such guarantee.

## 21. Who stayed infectious?

Five cells were infectious at step 12. The outlines identify the three that were still infectious at step 16. With irreversible recovery and independent recovery trials, a cell infectious now remains infectious four steps later with probability $(1-\theta_2)^4$. Counts alone discard those cell identities.

For the full experiment, pool the number infectious in both consecutive visits over all seven visit pairs and divide by the sum of the first seven infectious counts. For this recording 11 of 30 such cell-visits persist. Append that fraction to the eight counts. This yields a nine-number summary without access to hidden states.

## 22. What does the summary keep?

For the same eight observed maps, rejection ABC keeps the 64 closest simulations out of 8,192 eight-step-delay outbreaks. Distances are Euclidean after standardizing each summary coordinate across simulations. Counts alone give a central 90% recovery interval of $[0.162,0.347]$. Including persistence gives $[0.169,0.311]$. Both contain the generating value of about $0.265$.

## 23. Research Direction III: Model Misspecification

Up to this point, training and observation used the same simulator mechanism. Hold the recorded outbreak fixed and vary the simulator used to train the estimators.

## 24. When the truth lies outside the model

The blob is a set of possible **distributions**, not a region of parameter space. Each parameter indexes a distribution inside it. Model misspecification means $p_0\notin\mathcal P$. In the separated example shown, even the best candidate has positive discrepancy. In general a truth outside an unclosed family can have infimum zero. The dotted gap illustrates the separated case; its length is schematic.

## 25. Introducing misspecification to SIR

In this example a newly infected cell spends eight steps in the hidden exposed state before it becomes infectious. Visits every fourth step record only infectious cells. The matched simulator includes the exposed state. The misspecified simulator lets a newly infected cell become infectious immediately.

## 26. Simulating delayed infectiousness

Start at step 8 with the eight-step delay. The visit records zero infectious people while the state map still contains exposed people. Advance to step 12 to see five infectious people at the next visit. Switch to immediate onset at the same step to compare the two simulators with the same transmission and recovery probabilities, initial infections, and random draws. Reset returns to step 8 of the recorded outbreak.

## 27. The effect of misspecification

Recall the earlier visits. Step 8 has zero infectious cells and step 12 has five. Exposed cells can account for this reappearance. In the immediate-onset SIR model, extinction is absorbing, so the entire recording has zero likelihood for every parameter. The neural outputs in the bottom row are extrapolations at an observation outside the simulator’s support.

Read across the top row to compare the eight-step MCMC reference with the spline NPE and flow matching trained on eight-step outbreaks. The lower row repeats that reference so the change in each neural estimate is visible on the same axes. The six panels share a natural-log posterior density scale on the unit square. The darkest color includes all log densities at or below $-4$, including empty bins. The solid and dashed contours enclose approximately 50% and 90% of each smoothed density. The figure uses the archived plotting rule, a 40-by-40 histogram with a Gaussian filter of 1.5 bins. NPE receives the eight binary maps flattened into one input vector. Flow matching receives them as eight channels of a 2D convolutional network. Each training condition uses 8,192 training simulations, with 250 additional simulations for validation.

The recording contains 23 distinct cells observed infectious. The reference posterior has central 90% intervals of $[0.469,0.818]$ for transmission and $[0.184,0.304]$ for recovery. The generating values are about $(0.499,0.265)$.

## 28. Joint highest posterior density regions

At 0.90 on the horizontal axis, picture the contour enclosing the densest 90% of the fitted joint posterior on the transmission–recovery square. For each outbreak, check whether its generating pair falls inside that contour. The vertical axis shows the fraction of outbreaks covered. A curve below the diagonal indicates undercoverage.

Both saved spline-flow NPEs receive the same 192 held-out eight-step-delay outbreaks, with generating parameters drawn from the prior. For each outbreak, evaluate the joint log density at the generating pair and at 512 draws from that fit’s posterior on the unit square. The fraction of draws with higher density estimates the posterior mass above the generating pair’s density. Count the pair as covered when this fraction is at most the nominal mass. At 90%, the matched fit covered 153 of 192 pairs (79.7%) and the immediate-onset fit covered 48 (25.0%). The shading gives a pointwise 95% binomial reference range under calibration, accounting for the finite number of posterior draws.

## 29. Dealing with model misspecification

For a posterior ensemble, train several estimators with different random seeds and average their densities at the same observation. If the fits disagree, their mixture can express more uncertainty. All fits still share the simulator’s assumptions. [Cannon et al. (2022), Section 2.3](https://arxiv.org/pdf/2209.01845#page=6)

To learn robust summaries, train the summary network to retain information about $\theta$ while penalizing discrepancies between summaries of simulated and observed data. This reduces reliance on features the simulator reproduces poorly. [Huang et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/16c5b4102a6b6eb061e502ce6736ad8a-Paper-Conference.pdf)

An explicit discrepancy model specifies how an observation $x^*$ could differ from a simulator output $x_{\mathrm{sim}}$. Ward and colleagues introduce an error model $p(x^*\mid x_{\mathrm{sim}})$ and average over plausible simulator outputs when inferring $\theta$. [Ward et al. (2022)](https://papers.neurips.cc/paper_files/paper/2022/hash/db0eac6747e3631eb91095cd76065611-Abstract-Conference.html)

## 30. Research Direction IV: New advances

Ask which part of the workflow could be shared across inference tasks. One direction is to reuse a pretrained conditional predictor.

## 31. NPE-PFN

Vetter and colleagues use a pretrained tabular foundation model, TabPFN, for conditional density estimation. Simulated parameter–data pairs serve as context. For multiple parameters, conditional predictions can be combined autoregressively. The reported simulations and scientific benchmarks explore whether this reduces task-specific fitting and simulation costs. The SIR plots here use the existing NPE and FMPE fits; they are not results for NPE-PFN. [Vetter et al., NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/770cabd044c4eacb6dc5924d9a686dce-Abstract-Conference.html)

## 32. Conclusion

Return to the opening problem. A posterior about transmission and recovery is meaningful only in relation to the simulator, what the visits measured and what the estimator retained.

## 33. What determines the posterior?

Omitting the hidden delay changed the inferred transmission rate for the same recording. Keeping cell persistence sharpened recovery inference. Repeated outbreaks made the cost of misspecification visible. NPE-PFN offers another way to use simulated pairs; choosing a model, preserving information and checking uncertainty still require scientific judgment.

### Reproduction and provenance

The [outbreak selection](selection.json) records the generating parameters and random seed. All single-outbreak figures use the same eight observations. Each matched neural estimator used 8,192 training simulations and 250 validation simulations. The misspecified estimators used the same simulation budget with immediate onset.

To rebuild the standalone slides from the included figures and playback data, run `quarto render slides.qmd` in this directory.

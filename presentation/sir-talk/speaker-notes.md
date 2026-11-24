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

Ask the two questions separately. Transmission refers to a susceptible person in contact with an infectious neighbor. In this SIR simulator a newly infected person is infectious at the next step. Recovery refers to someone who is already infectious. A snapshot alone can confound a large number of new infections with a long infectious period.

## 8. Let’s make a simulator

Press **Play**, then pause between two visits. The hidden states on the left can change while the latest recorded map on the right stays fixed. A zero on the right can indicate a susceptible or recovered person at the last visit. Change the sliders and replay a few steps. Reset restores the original parameter pair and returns the SIR simulation to step 8. The same random stream makes the settings easy to compare. The grid has four-neighbor contact, irreversible recovery and synchronous updates. With $k$ infectious neighbors the per-step infection probability is $1-(1-\theta_1)^k$.

## 9. Real-world observations

These maps show the introductory SIR simulation at its default parameters. Visits every four steps record which cells are infectious. The eight counts are 25, 62, 90, 118, 114, 91, 38 and 10. The estimator receives the maps. Adjusting the sliders on the previous slide leaves this observation fixed.

## 10. Neural Methods for SBI

We can generate supervised training examples: on each simulation the generating parameter is known. The two methods use the same kind of pairs and reverse what they predict.

## 11. Conditional density estimation — NLE

Point across the diagram. Each superscript identifies a whole simulation, while subscripts on $\theta$ identify its two components. Use a neural network to parameterize the conditional density $q_\phi$. Here $\phi$ denotes the network weights and $\theta$ denotes the simulator parameters. Reveal the loss and minimize it over $\phi$. For each simulated pair, score the density assigned to the simulated observation given its generating parameter. Training minimizes the average negative log density across these pairs. After training, evaluate it at the fixed observation and proposed parameter values. Multiply the learned likelihood by the prior to get an approximate MCMC target.

## 12. Conditional density estimation — NPE

Advance from the crossed-out NLE label to NPE. The approximation and loss reverse direction together. Fit the posterior network’s weights $\phi$ by minimizing $\mathcal L_{\mathrm{NPE}}(\phi)$. Use the same simulated pairs, and now score the density assigned to the generating parameter given its observation. Both expectations average over the joint draw shown at the top of the slide. With training parameters drawn from the prior, the conditional distribution of the generating parameter given a simulated observation is the model posterior. At presentation time, give the fitted estimator the eight observed binary maps and draw parameter pairs directly.

## 13. Transport maps

The fitted examples from this point use the archived outbreak with an eight-step delay and the same generating parameter pair as the introductory SIR run. The left heatmap shows the standard Gaussian base density. The right heatmap estimates the fitted SIR flow’s joint density for transmission and recovery using 16,384 draws conditional on the eight observed binary maps. Each panel scales its color to its own peak. The axes give Gaussian coordinates on the left and probability units on the right. The flow-matching slide animates individual trajectories under this same fit.

## 14. Normalizing flows

Hold $x$ fixed and invert the map to recover the Gaussian input that produced $\theta$. The determinant adjusts the base density for the local change in volume. Figure 1 of [Dax et al. (2023)](https://arxiv.org/pdf/2305.17161#page=2) illustrates a discrete flow for a two-moons example. The observation conditions each layer. Requiring efficient inversion and determinant evaluation restricts the choice of layers.

## 15. Research Direction I: Generative Modeling for SBI

How can we learn a transport map for the posterior by fitting a velocity field?

## 16. Flow matching

Draw a simulation pair $(\theta,x)$, independent Gaussian noise $\eta$, and a uniform time $t$. The straight interpolation between the noise and the parameter has known velocity $\theta-\eta$. Squared-error regression teaches the network the average target velocity conditional on location, time, and observation. To sample for a new observation $x^*$, initialize a fresh particle at $\eta$ and integrate $\dot\theta_t=v_{t,x^*}(\theta_t)$. The training path uses the actual generating parameter pair as its endpoint. A two-dimensional convolutional encoder takes the eight recorded maps as channels. Its output conditions the velocity field. The FMPE panel is the paper’s two-moons example.

## 17. Score matching

Point along the top arrow, then back along the bottom. “The forward diffusion adds noise. To sample, we start from noise and use the learned score to work back to a parameter draw for these outbreak maps.”

The training pairs follow $p(\theta_0)p(x\mid\theta_0)$. Draw $t$ uniformly on $(0,T)$ and $\theta_t$ from the known Gaussian noising kernel $p_t(\cdot\mid\theta_0)$. The gradient in the loss is available from that kernel. Its conditional mean given $(\theta_t,x,t)$ is the noisy posterior score. The diagram shows the reverse SDE. The same score also defines a probability-flow ODE whose velocity is $f(\theta_t,t)-\tfrac12g(t)^2s_{t,x}(\theta_t)$, integrated from $T$ to $0$. [Sharrock et al., Fig. 1](https://proceedings.mlr.press/v235/sharrock24a.html) and [Chen, Bansal & Scott, Eqs. 1–2](https://arxiv.org/html/2410.19105v3#S3)

## 18. Sampling and density evaluation

Both computations hold the observation fixed. For sampling, start from Gaussian noise and follow the field forward. For density evaluation, start at the query point and trace the trajectory back to its Gaussian starting point. Along the trajectory, positive divergence means a small volume expands and its log density decreases. This gives the minus sign in the integral from zero to one. [Dax et al. (2023), Eqs. (2)–(3)](https://arxiv.org/pdf/2305.17161#page=4)

## 19. Flow matching in the SIR example

Press **Play**. Each orange dot starts as an independent standard Gaussian draw and follows the fitted conditional velocity field in the same two-dimensional space. The neural network was trained on the simulator’s transmission and recovery parameters. With the spatial features of the eight observed maps held fixed, integrate each particle from time zero to one using the midpoint rule. At time one the two components are the estimated transmission and recovery probabilities. Earlier states are intermediate ODE values and can be outside the unit square. White arrows show network evaluations at the current time. The background estimates the current density of 16,384 particles. Both panels have fixed axes throughout playback. The right panel shows the full unit square outlined on the left, with the same particles and relative density scale. The final heatmap is shared with Transport maps.

## 20. Recent advances

These methods let us combine evidence across independent outbreaks with shared parameters, draw posterior samples in fewer steps, or use calibration pairs to correct a simulator mismatch.

## 21. Research Direction II: Summary Statistics

Return to the observation side of the same experiment. What could we retain from eight maps if we had room for only a small vector?

## 22. From eight maps to eight counts

Read the row left to right. Each number counts red cells directly above it. We compress $8\times24\times24=4{,}608$ binary measurements to eight numbers. Here we explore what would happen if we compressed the maps. In the ABC example, apply the same transformation to simulated and observed maps, then match on the summaries. It recovers the full-data posterior if the summary is sufficient for these parameters; arbitrary compression gives no such guarantee.

## 23. Who stayed infectious?

Five cells were infectious at step 12. The outlines identify the three that were still infectious at step 16. With irreversible recovery and independent recovery trials, a cell infectious now remains infectious four steps later with probability $(1-\theta_2)^4$. Counts alone discard those cell identities.

For the full experiment, pool the number infectious in both consecutive visits over all seven visit pairs and divide by the sum of the first seven infectious counts. For this recording 11 of 30 such cell-visits persist. Append that fraction to the eight counts. This yields a nine-number summary without access to hidden states.

## 24. What does the summary keep?

For the same eight observed maps, rejection ABC keeps the 64 closest simulations out of 8,192 eight-step-delay outbreaks. Distances are Euclidean after standardizing each summary coordinate across simulations. Counts alone give a central 90% recovery interval of $[0.162,0.347]$. Including persistence gives $[0.169,0.311]$. Both contain the generating value of about $0.265$.

## 25. Research Direction III: Model Misspecification

Up to this point, training and observation used the same simulator mechanism. Hold the recorded outbreak fixed and vary the simulator used to train the estimators.

## 26. When the truth lies outside the model

The blob is a set of possible **distributions**, not a region of parameter space. Each parameter indexes a distribution inside it. Model misspecification means $p_0\notin\mathcal P$. In the separated example shown, even the best candidate has positive discrepancy. In general a truth outside an unclosed family can have infimum zero. The dotted gap illustrates the separated case; its length is schematic.

## 27. Introducing misspecification to SIR

The inference model assumes immediate onset. To generate the observation in the misspecified case, insert an eight-step exposed period before infectiousness. Both outbreaks start with the same four infectious cells and use the same transmission and recovery probabilities and random draws.

## 28. Simulating a four-step delay

Newly infected cells spend four steps exposed before becoming infectious. The right panel shows the infectious-cell maps recorded at steps 8, 12 and 16.

## 29. The effect of misspecification

Each row uses two infectious-cell maps, recorded at steps 9 and 10, at the same generating pair of approximately $(0.499, 0.265)$. The immediate-onset outbreak has 55 and 62 infectious cells. The eight-step-delay outbreak has four and six. Both neural estimators trained on immediate-onset outbreaks. Crosses mark the generating pair, circles mark posterior means, and the solid and dashed contours enclose 50% and 90% joint posterior mass.

The reference column uses MCMC for immediate onset and the exact delayed-model likelihood for the eight-step delay. On the delayed maps, the reference mean is $(0.518, 0.262)$. Spline NPE gives $(0.104, 0.184)$ and flow matching gives $(0.110, 0.182)$. A separate immediate-onset MCMC fit to those same maps gives $(0.097, 0.149)$, confirming the direction of both shifts for this selected example.

## 30. Coverage under delayed onset

Keep the immediate-onset estimators fixed. For each delay, use the same 192 parameter pairs drawn from the uniform prior and the same simulator seeds. Each outbreak contributes one infectious-cell map at step 12.

For the 90% region, draw 1,024 parameter pairs from the fitted posterior conditional on the map. Take the 10th percentile of their fitted densities. Count the outbreak if the fitted density at its generating pair reaches or exceeds that cutoff. Divide by 192. Repeat with the corresponding cutoff for each nominal mass.

## 31. Dealing with model misspecification

For a posterior ensemble, train several estimators with different random seeds and average their densities at the same observation. If the fits disagree, their mixture can express more uncertainty. All fits still share the simulator’s assumptions. [Cannon et al. (2022), Section 2.3](https://arxiv.org/pdf/2209.01845#page=6)

To learn robust summaries, train the summary network to retain information about $\theta$ while penalizing discrepancies between summaries of simulated and observed data. This reduces reliance on features the simulator reproduces poorly. [Huang et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/16c5b4102a6b6eb061e502ce6736ad8a-Paper-Conference.pdf)

An explicit discrepancy model specifies how an observation $x^*$ could differ from a simulator output $x_{\mathrm{sim}}$. Ward and colleagues introduce an error model $p(x^*\mid x_{\mathrm{sim}})$ and average over plausible simulator outputs when inferring $\theta$. [Ward et al. (2022)](https://papers.neurips.cc/paper_files/paper/2022/hash/db0eac6747e3631eb91095cd76065611-Abstract-Conference.html)

## 32. New advances

The first slide asks how much of an inference procedure we can reuse. The second asks what we can learn from the hidden steps in a simulation.

## 33. Research Direction IV: Scalable and flexible SBI

For the outbreak, one inference question fixes the maps and asks about transmission and recovery. Another fixes the parameters and asks what maps might be observed. Simformer can approximate both conditionals with one trained network by changing which variables it conditions on. [Gloeckler et al. (2024)](https://proceedings.mlr.press/v235/gloeckler24a.html)

NPE-PFN reuses a pretrained tabular predictor. The simulated pairs form its context, and the observed data are the query. To draw several parameters jointly, it can sample each parameter conditional on the ones already drawn. [Vetter et al. (2025)](https://papers.nips.cc/paper_files/paper/2025/hash/770cabd044c4eacb6dc5924d9a686dce-Abstract-Conference.html)

## 34. Research Direction V: Simulator transparency

In the outbreak, $x$ contains the infectious-cell maps at eight visits. The simulator also knows each cell’s state between visits. Those hidden states are part of $z$.

When the simulator exposes the probabilities of its internal transitions, its joint score $\nabla_\theta\log p(x,z\mid\theta)$ provides a training target. Averaging that score conditional on $x$ gives the likelihood score. [Brehmer et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7071889/)

Inference compilation uses recorded simulation paths to learn a proposal for each successive random choice given the observations and previous choices. Importance sampling then weights the proposed paths. [Le et al. (2017)](https://proceedings.mlr.press/v54/le17a.html)

## Thank you

Special thanks to Yichen Ji and Prof. Ricardo Baptista.

### Reproduction and provenance

The [outbreak selection](selection.json) records the generating parameters and random seed. The introductory observation uses the immediate-onset SIR replay at those parameters. The transport and summary examples use the archived eight-step recording. The posterior comparison pairs immediate-onset and eight-step-delay recordings at seed 3077963830 and the same generating parameters. Only visits at steps 9 and 10 enter this comparison. Its two-map estimators used 8,192 immediate-onset training outbreaks and 250 validation outbreaks.

To rebuild the standalone slides from the included figures and playback data, run `quarto render slides.qmd` in this directory.

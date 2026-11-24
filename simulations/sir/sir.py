"""The spatial SIR simulator used in the presentation."""

import numpy as np

GRID_SIZE = 24
VISITS = (4, 8, 12)
SUSCEPTIBLE, INFECTIOUS, RECOVERED = 0, 1, 2

# The presentation uses zero-based grid positions and four central infections.
OBSERVED_THETA = (0.41, 0.18)
OBSERVED_SEED = 917
SUMMARY_NAMES = ("C4", "C8", "C12", "O4,8", "O8,12")


def count_infectious_neighbors(infectious):
    """For each cell, count infectious neighbors up, down, left, and right.

    Shape stays (number of simulations, rows, columns). The slices stop at
    the edges, so the left and right borders are not neighbors.
    """
    neighbors = np.zeros_like(infectious, dtype=np.uint8)
    neighbors[:, 1:, :] += infectious[:, :-1, :]  # neighbor above
    neighbors[:, :-1, :] += infectious[:, 1:, :]  # neighbor below
    neighbors[:, :, 1:] += infectious[:, :, :-1]  # neighbor to the left
    neighbors[:, :, :-1] += infectious[:, :, 1:]  # neighbor to the right
    return neighbors


def sir_step(states, transmission, recovery, rng):
    """Advance all grids by one step using their states from the same time."""
    infectious = states == INFECTIOUS
    neighbors = count_infectious_neighbors(infectious)

    # A susceptible person escapes infection only if all k contacts fail.
    infection_chance = 1 - (1 - transmission) ** neighbors
    infection_draw = rng.random(states.shape)
    recovery_draw = rng.random(states.shape)

    new_infections = (states == SUSCEPTIBLE) & (infection_draw < infection_chance)
    new_recoveries = infectious & (recovery_draw < recovery)
    next_states = states.copy()
    next_states[new_infections] = INFECTIOUS
    next_states[new_recoveries] = RECOVERED
    return next_states


def simulate_visits(parameters, rng):
    """Return infectious masks of shape (batch, 3, 24, 24).

    Each row in parameters is [p, q]. The output contains only the three
    visits. True and False mean infectious and not infectious, respectively.
    """
    parameters = np.atleast_2d(np.asarray(parameters, dtype=float))
    if parameters.ndim != 2 or parameters.shape[1] != 2:
        raise ValueError("parameters must have shape (batch, 2)")
    if not np.all((0 <= parameters) & (parameters <= 1)):
        raise ValueError("p and q must be probabilities between zero and one")

    transmission = parameters[:, 0, None, None]
    recovery = parameters[:, 1, None, None]
    states = np.full(
        (len(parameters), GRID_SIZE, GRID_SIZE), SUSCEPTIBLE, dtype=np.uint8
    )
    states[:, 11:13, 11:13] = INFECTIOUS

    visited = []
    for step in range(1, VISITS[-1] + 1):
        states = sir_step(states, transmission, recovery, rng)
        if step in VISITS:
            visited.append(states == INFECTIOUS)
    return np.stack(visited, axis=1)


def summarize(masks):
    """Return [C4, C8, C12, O4,8, O8,12] for each simulated dataset."""
    counts = masks.sum(axis=(-2, -1))
    # AND keeps the same cell only if it is infectious at both visits.
    overlaps = (masks[:, :-1] & masks[:, 1:]).sum(axis=(-2, -1))
    return np.concatenate([counts, overlaps], axis=1)


def simulate_summaries(parameters, rng, batch_size=512):
    """Keep at most one batch of grids in memory while building the dataset."""
    summaries = []
    for start in range(0, len(parameters), batch_size):
        batch = parameters[start : start + batch_size]
        summaries.append(summarize(simulate_visits(batch, rng)))
    return np.concatenate(summaries)

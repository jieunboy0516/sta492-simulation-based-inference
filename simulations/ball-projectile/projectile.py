"""The ball physics and the probability model, without any neural networks.

Angles in the public functions are in DEGREES. Distances are in metres.
Wind means a constant horizontal acceleration, not an air speed.
"""

import numpy as np
from scipy.integrate import quad_vec
from scipy.stats import norm, truncnorm


SPEED = 13.0             # initial speed, metres / second
GRAVITY = 9.81          # downward acceleration, metres / second**2
WIND_RATE = 2.0         # exponential rate; mean wind acceleration = 1 / rate
MEASUREMENT_SD = 0.25   # standard deviation of distance measurement, metres
PRIOR_MEAN = 35.0       # degrees, before truncation
PRIOR_SD = 15.0         # degrees, before truncation
ANGLE_LIMIT = 90.0      # degrees


def sample_prior(num_samples, rng):
    """Draw angles from Normal(35, 15**2), restricted to (0, 90) degrees.

    The result has shape (num_samples, 1): one unknown parameter per row.
    rng is a NumPy random generator, passed explicitly for reproducibility.
    """
    lower = (0.0 - PRIOR_MEAN) / PRIOR_SD
    upper = (ANGLE_LIMIT - PRIOR_MEAN) / PRIOR_SD
    angles = truncnorm.rvs(
        lower, upper, loc=PRIOR_MEAN, scale=PRIOR_SD,
        size=num_samples, random_state=rng,
    )
    return angles[:, None]


def prior_density(angle_deg):
    """Prior density PER DEGREE; its integral from 0 to 90 is one."""
    return truncnorm.pdf(
        angle_deg, -PRIOR_MEAN / PRIOR_SD,
        (ANGLE_LIMIT - PRIOR_MEAN) / PRIOR_SD,
        loc=PRIOR_MEAN, scale=PRIOR_SD,
    )


def flight_time(angle_deg):
    """The positive root of height(t) = 0, after leaving the ground."""
    angle_deg = np.asarray(angle_deg, dtype=np.float64)
    if np.any((angle_deg < 0) | (angle_deg > ANGLE_LIMIT)):
        raise ValueError("Angles must be between 0 and 90 degrees.")
    angle_rad = np.deg2rad(angle_deg)
    return 2.0 * SPEED * np.sin(angle_rad) / GRAVITY


def range_coefficients(angle_deg):
    """Return A and B so the noise-free landing distance is A + B * wind."""
    angle_rad = np.deg2rad(np.asarray(angle_deg, dtype=np.float64))
    duration = flight_time(angle_deg)
    no_wind_range = SPEED * np.cos(angle_rad) * duration
    wind_coefficient = 0.5 * duration**2
    return no_wind_range, wind_coefficient


def trajectory(angle_deg, wind_acceleration=0.0, num_points=150):
    """Return time, horizontal position, and height for one noise-free flight.

    We can calculate the trajectory directly; no ODE solver or time stepping
    is needed. Measurement noise affects the recorded range, not this path.
    """
    if wind_acceleration < 0:
        raise ValueError("This model uses nonnegative tailwind acceleration.")
    angle_rad = np.deg2rad(angle_deg)
    duration = float(flight_time(angle_deg))
    time = np.linspace(0.0, duration, num_points)
    horizontal = SPEED * np.cos(angle_rad) * time + 0.5 * wind_acceleration * time**2
    height = SPEED * np.sin(angle_rad) * time - 0.5 * GRAVITY * time**2
    return time, horizontal, height


def simulate(angle_deg, rng):
    """One random throw per angle; return measured ranges with shape (N, 1).

    The simulator resamples both wind and measurement noise on EVERY call.
    Wind is an unobserved noise source, not a network input or inference target.
    """
    angles = np.asarray(angle_deg, dtype=np.float64).reshape(-1)
    no_wind_range, wind_coefficient = range_coefficients(angles)
    wind = rng.exponential(scale=1.0 / WIND_RATE, size=len(angles))
    measurement_noise = rng.normal(0.0, MEASUREMENT_SD, size=len(angles))
    measured_range = no_wind_range + wind_coefficient * wind + measurement_noise
    # Negative measurements are possible under Gaussian noise; do not clip them.
    return measured_range[:, None]


# The functions below are for CHECKING inference, not for training the network.
def reference_likelihood(distance_m, angle_deg):
    """Numerically integrate out wind to evaluate p(distance | angle).

    Substitute q = WIND_RATE * wind, so the exponential density is exp(-q).
    quad_vec integrates all supplied angles together using adaptive quadrature.
    It returns a numerical error estimate as well as the likelihood values.
    """
    no_wind_range, wind_coefficient = range_coefficients(angle_deg)

    def integrand(q):
        expected_range = no_wind_range + wind_coefficient * q / WIND_RATE
        return np.exp(-q) * norm.pdf(
            distance_m, loc=expected_range, scale=MEASUREMENT_SD
        )

    likelihood, error = quad_vec(
        integrand, 0.0, np.inf, epsabs=1e-11, epsrel=1e-9
    )
    return likelihood, float(error)


def reference_posterior(distance_m, grid_size=4000):
    """Bayes' rule on a midpoint angle grid covering the whole prior support."""
    step = ANGLE_LIMIT / grid_size
    angle_deg = (np.arange(grid_size) + 0.5) * step
    likelihood, quadrature_error = reference_likelihood(distance_m, angle_deg)
    unnormalized = prior_density(angle_deg) * likelihood
    evidence = np.sum(unnormalized) * step
    if not np.isfinite(evidence) or evidence <= 0:
        raise ValueError("Observation is numerically outside the reference calculation.")
    density = unnormalized / evidence
    return {
        "angle_deg": angle_deg,
        "density": density,
        "step": step,
        "evidence": float(evidence),
        "quadrature_error": quadrature_error,
        "mean_deg": float(np.sum(angle_deg * density) * step),
        "low_angle_probability": float(np.sum(density[angle_deg < 45.0]) * step),
    }

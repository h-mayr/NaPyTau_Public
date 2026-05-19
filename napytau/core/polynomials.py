from napytau.core.errors.polynomial_coefficient_error import (
    PolynomialCoefficientError,
)
import numpy as np
import scipy as sp
from scipy.interpolate import make_lsq_spline, make_splrep, BSpline

from napytau.core.time import calculate_times_from_distances_and_relative_velocity
from napytau.import_export.model.dataset import DataSet


def evaluate_polynomial_at_measuring_times(
    dataset: DataSet,
    coefficients: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Evaluates a B-spline at the measuring time points.

    Args:
        dataset (DataSet): The dataset of the experiment
        coefficients (ndarray): B-spline coefficients (.c from fitted spline)
        knots (ndarray): Full B-spline knot sequence (.t from fitted spline)
        degree (int): Degree of the B-spline (k parameter)

    Returns:
        ndarray: Array of B-spline values evaluated at the given time points.
    """
    if len(coefficients) == 0:
        raise PolynomialCoefficientError(
            "An empty array of coefficients can not be evaluated."
        )

    times: np.ndarray = calculate_times_from_distances_and_relative_velocity(dataset)
    return np.asarray(BSpline(knots, coefficients, degree)(times))


def evaluate_differentiated_polynomial_at_measuring_times(
    dataset: DataSet,
    coefficients: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Evaluates the derivative of a B-spline at the measuring time points.

    Args:
        dataset (DataSet): The dataset of the experiment
        coefficients (ndarray): B-spline coefficients (.c from fitted spline)
        knots (ndarray): Full B-spline knot sequence (.t from fitted spline)
        degree (int): Degree of the B-spline (k parameter)

    Returns:
        ndarray: Array of B-spline derivative values at the given time points.
    """
    if len(coefficients) == 0:
        raise PolynomialCoefficientError(
            "An empty array of coefficients can not be evaluated."
        )

    times: np.ndarray = calculate_times_from_distances_and_relative_velocity(dataset)
    return np.asarray(BSpline(knots, coefficients, degree).derivative()(times))


def calculate_polynomial_coefficients_for_fit(
    dataset: DataSet,
    degree: int,
    smoothing_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits a B-spline to the dataset and returns the B-spline coefficients
    and the full knot sequence.

    When smoothing_factor is None (default), uses make_lsq_spline with
    interior knots from dataset.get_sampling_points().
    When smoothing_factor is a float, uses make_splrep which selects
    knots automatically via the smoothing parameter s.

    Args:
        dataset (DataSet): The dataset of the experiment
        degree (int): The degree of the spline (k parameter)
        smoothing_factor (float | None): If None, use LSQ mode; if float, use
            make_splrep with this smoothing parameter s.

    Returns:
        tuple[ndarray, ndarray]: (B-spline coefficients, full knot sequence)
    """
    times: np.ndarray = calculate_times_from_distances_and_relative_velocity(dataset)
    shifted_intensities = np.array(
        dataset.get_datapoints().get_normalized_shifted_intensities().get_values()
    )

    # Both spline variants require data sorted in ascending x order
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    shifted_intensities = shifted_intensities[sort_idx]

    if smoothing_factor is not None:
        spline = make_splrep(times, shifted_intensities, k=degree, s=smoothing_factor)
        return np.asarray(spline.c), np.asarray(spline.t)

    sampling_points = dataset.get_sampling_points()

    # Interior knots must lie strictly inside (times[0], times[-1])
    t_min, t_max = times[0], times[-1]
    interior_knots = (
        np.array(sorted(t for t in sampling_points if t_min < t < t_max))
        if sampling_points
        else np.array([])
    )

    if len(interior_knots) == 0:
        # No valid knots: fall back to make_splrep with automatic smoothing
        spline = make_splrep(times, shifted_intensities, k=degree)
        return np.asarray(spline.c), np.asarray(spline.t)

    t_full = np.concatenate(
        [[times[0]] * (degree + 1), interior_knots, [times[-1]] * (degree + 1)]
    )
    spline = make_lsq_spline(times, shifted_intensities, t=t_full, k=degree)
    return np.asarray(spline.c), np.asarray(spline.t)


def calculate_polynomial_coefficients_for_coupled_fit(
    dataset: DataSet,
    tau_factor: float,
    degree: int,
    weight_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits B-spline coefficients using a coupled system where both shifted and
    unshifted intensities enter the normal equations with tau_factor as a
    parameter (mirrors the Perl napatau calc_fit routine).

    Args:
        dataset (DataSet): The dataset of the experiment
        tau_factor (float): Current tau factor estimate
        degree (int): Degree of the B-spline
        weight_factor (float): Weight for unshifted contribution (default 1.0)

    Returns:
        tuple[ndarray, ndarray]: (B-spline coefficients, full knot sequence)
    """
    times: np.ndarray = calculate_times_from_distances_and_relative_velocity(dataset)
    datapoints = dataset.get_datapoints()

    i_sh = np.array(datapoints.get_normalized_shifted_intensities().get_values())
    sigma_sh = np.array(datapoints.get_normalized_shifted_intensities().get_errors())
    i_us = np.array(datapoints.get_normalized_unshifted_intensities().get_values())
    sigma_us = np.array(datapoints.get_normalized_unshifted_intensities().get_errors())

    sort_idx = np.argsort(times)
    times = times[sort_idx]
    i_sh = i_sh[sort_idx]
    sigma_sh = sigma_sh[sort_idx]
    i_us = i_us[sort_idx]
    sigma_us = sigma_us[sort_idx]

    sampling_points = dataset.get_sampling_points()
    t_min, t_max = times[0], times[-1]
    interior_knots = (
        np.array(sorted(t for t in sampling_points if t_min < t < t_max))
        if sampling_points
        else np.array([])
    )

    if len(interior_knots) == 0:
        ref_spline = make_splrep(times, i_sh, k=degree)
        knots = np.asarray(ref_spline.t)
    else:
        t_full = np.concatenate(
            [[times[0]] * (degree + 1), interior_knots, [times[-1]] * (degree + 1)]
        )
        ref_spline = make_lsq_spline(times, i_sh, t=t_full, k=degree)
        knots = np.asarray(ref_spline.t)

    n = len(times)
    n_coeffs = len(knots) - degree - 1

    b_matrix = np.zeros((n, n_coeffs))
    db_matrix = np.zeros((n, n_coeffs))
    for j in range(n_coeffs):
        e_j = np.zeros(n_coeffs)
        e_j[j] = 1.0
        spl = BSpline(knots, e_j, degree)
        b_matrix[:, j] = np.asarray(spl(times))
        db_matrix[:, j] = np.asarray(spl.derivative()(times))

    fac1 = (2.0 - weight_factor) / sigma_sh**2
    fac2 = weight_factor * tau_factor / sigma_us**2

    a_matrix = (
        b_matrix.T @ np.diag(fac1) @ b_matrix
        + db_matrix.T @ np.diag(fac2 * tau_factor) @ db_matrix
    )
    y_vector = b_matrix.T @ (fac1 * i_sh) + db_matrix.T @ (fac2 * i_us)

    coefficients, _, _, _ = np.linalg.lstsq(a_matrix, y_vector, rcond=None)

    return np.asarray(coefficients), knots


def calculate_polynomial_coefficients_for_tau_factor(
    dataset: DataSet,
    tau_factor: float,
    degree: int,
    smoothing_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds B-spline coefficients such that P(t)/P'(t) = tau_factor at all
    measuring time points, using the knot structure from a reference spline fit.

    Args:
        dataset (DataSet): The dataset of the experiment
        tau_factor (float): The tau factor to be used in the spline fit
        degree (int): The degree of the spline
        smoothing_factor (float | None): Passed to
            calculate_polynomial_coefficients_for_fit.

    Returns:
        tuple[ndarray, ndarray]: (optimized B-spline coefficients, full knot sequence)
    """
    reference_coefficients, knots = calculate_polynomial_coefficients_for_fit(
        dataset, degree, smoothing_factor
    )

    times: np.ndarray = calculate_times_from_distances_and_relative_velocity(dataset)

    def residuals(c: np.ndarray) -> np.ndarray:
        spline = BSpline(knots, c, degree)
        return np.asarray(
            spline(times) / spline.derivative()(times) - tau_factor
        )

    res = sp.optimize.least_squares(residuals, reference_coefficients)

    return np.array(res.x), knots

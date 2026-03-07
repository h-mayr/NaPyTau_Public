from napytau.core.errors.polynomial_coefficient_error import (
    PolynomialCoefficientError,
)
import numpy as np
import scipy as sp
from scipy.interpolate import LSQUnivariateSpline, BSpline

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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits a B-spline to the dataset using LSQUnivariateSpline and returns
    the B-spline coefficients and the full knot sequence.

    Interior knots come from dataset.get_sampling_points().

    Args:
        dataset (DataSet): The dataset of the experiment
        degree (int): The degree of the spline (k parameter)

    Returns:
        tuple[ndarray, ndarray]: (B-spline coefficients, full knot sequence)
    """
    sampling_points = dataset.get_sampling_points()
    if not sampling_points:
        raise ValueError(
            "dataset.sampling_points must be non-empty for LSQUnivariateSpline fitting."
        )

    times: np.ndarray = calculate_times_from_distances_and_relative_velocity(dataset)
    shifted_intensities = (
        dataset.get_datapoints().get_shifted_intensities().get_values()
    )
    interior_knots = np.array(sampling_points)

    spline = LSQUnivariateSpline(
        times, shifted_intensities, t=interior_knots, k=degree
    )
    return spline.c, spline.t


def calculate_polynomial_coefficients_for_tau_factor(
    dataset: DataSet,
    tau_factor: float,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds B-spline coefficients such that P(t)/P'(t) = tau_factor at all
    measuring time points, using the knot structure from a reference spline fit.

    Args:
        dataset (DataSet): The dataset of the experiment
        tau_factor (float): The tau factor to be used in the spline fit
        degree (int): The degree of the spline

    Returns:
        tuple[ndarray, ndarray]: (optimized B-spline coefficients, full knot sequence)
    """
    reference_coefficients, knots = calculate_polynomial_coefficients_for_fit(
        dataset, degree
    )

    times: np.ndarray = calculate_times_from_distances_and_relative_velocity(dataset)

    def residuals(c: np.ndarray) -> np.ndarray:
        spline = BSpline(knots, c, degree)
        return np.asarray(
            spline(times) / spline.derivative()(times) - tau_factor
        )

    res = sp.optimize.least_squares(residuals, reference_coefficients)

    return np.array(res.x), knots

from napytau.core.polynomials import (
    evaluate_differentiated_polynomial_at_measuring_times,
    evaluate_polynomial_at_measuring_times,
)
import numpy as np
import scipy as sp
from typing import Tuple

from napytau.import_export.model.dataset import DataSet


def calculate_chi_squared(
    dataset: DataSet,
    coefficients: np.ndarray,
    tau_factor: float,
    weight_factor: float,
    knots: np.ndarray,
    degree: int,
) -> float:
    """
    Computes the chi-squared value for a given hypothesis t_hyp

    Args:
        dataset (DataSet): The dataset of the experiment
        coefficients (ndarray): B-spline coefficients for fitting
        tau_factor (float): Hypothesis value for the scaling factor
        weight_factor (float): Weighting factor for unshifted intensities
        knots (ndarray): Full B-spline knot sequence
        degree (int): Degree of the B-spline

    Returns:
        float: The chi-squared value for the given inputs.
    """

    datapoints = dataset.get_datapoints()
    # Compute the difference between Doppler-shifted intensities and spline model
    shifted_intensity_difference: np.ndarray = (
        datapoints.get_shifted_intensities().get_values()
        - evaluate_polynomial_at_measuring_times(dataset, coefficients, knots, degree)
    ) / datapoints.get_shifted_intensities().get_errors()

    # Compute the difference between unshifted intensities and
    # scaled derivative of the spline model
    unshifted_intensity_difference: np.ndarray = (
        datapoints.get_unshifted_intensities().get_values()
        - (
            tau_factor
            * evaluate_differentiated_polynomial_at_measuring_times(
                dataset, coefficients, knots, degree
            )
        )
    ) / datapoints.get_unshifted_intensities().get_errors()

    # combine the weighted sum of squared differences
    result: float = np.sum(
        (np.power(shifted_intensity_difference, 2))
        + (weight_factor * (np.power(unshifted_intensity_difference, 2)))
    )

    return result


def optimize_tau_factor(
    dataset: DataSet,
    weight_factor: float,
    coefficients: np.ndarray,
    tau_factor_range: Tuple[float, float],
    knots: np.ndarray,
    degree: int,
) -> float:
    """
    Optimizes the hypothesis value t_hyp to minimize the chi-squared function.

    Parameters:
        dataset (DataSet): The dataset of the experiment
        weight_factor (float): Weighting factor for unshifted intensities
        coefficients (ndarray): B-spline coefficients for fitting
        tau_factor_range (tuple): Range for hypothesis optimization (min, max)
        knots (ndarray): Full B-spline knot sequence
        degree (int): Degree of the B-spline

    Returns:
        float: Optimized t_hyp value.
    """
    result: sp.optimize.OptimizeResult = sp.optimize.minimize(
        lambda t_hyp: calculate_chi_squared(
            dataset,
            coefficients,
            t_hyp,
            weight_factor,
            knots,
            degree,
        ),
        # Initial guess for t_hyp. Starting with the mean reduces likelihood of
        # biasing the optimization process toward one boundary.
        x0=np.ndarray(shape=(1,), buffer=np.array([np.mean(tau_factor_range)])),
        bounds=[(tau_factor_range[0], tau_factor_range[1])],
    )

    # Return optimized t_hyp value
    return float(result.x)

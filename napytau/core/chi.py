from napytau.core.polynomials import (
    calculate_polynomial_coefficients_for_coupled_fit,
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

    # Reduce by degrees of freedom so a perfect fit gives χ²_red ≈ 1
    n_data = len(shifted_intensity_difference)
    n_terms = n_data * (1 + int(weight_factor > 0))
    dof = max(n_terms - len(coefficients), 1)
    return result / dof


def optimize_tau_factor(
    dataset: DataSet,
    weight_factor: float,
    coefficients: np.ndarray,
    tau_factor_range: Tuple[float, float],
    knots: np.ndarray,
    degree: int,
    fit_mode: str = "lsq",
    initial_guess: float | None = None,
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
        fit_mode (str): "lsq", "smooth", or "coupled"

    Returns:
        float: Optimized t_hyp value.
    """

    def _objective(t_hyp: np.ndarray) -> float:
        tau = float(t_hyp[0]) if hasattr(t_hyp, "__len__") else float(t_hyp)
        if fit_mode == "coupled":
            c, k = calculate_polynomial_coefficients_for_coupled_fit(
                dataset, tau, degree
            )
        else:
            c, k = coefficients, knots
        return calculate_chi_squared(dataset, c, tau, weight_factor, k, degree)

    # Use provided guess; fall back to geometric mean (better than arithmetic
    # mean when the range spans multiple orders of magnitude).
    x0_val = (
        initial_guess
        if initial_guess is not None
        else float(np.sqrt(tau_factor_range[0] * tau_factor_range[1]))
    )
    x0_val = float(np.clip(x0_val, tau_factor_range[0], tau_factor_range[1]))

    result: sp.optimize.OptimizeResult = sp.optimize.minimize(
        _objective,
        x0=np.ndarray(shape=(1,), buffer=np.array([x0_val])),
        bounds=[(tau_factor_range[0], tau_factor_range[1])],
    )

    # Return optimized t_hyp value
    return float(result.x)

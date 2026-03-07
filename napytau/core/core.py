from napytau.core.chi import optimize_tau_factor
from napytau.core.polynomials import (
    calculate_polynomial_coefficients_for_fit,
    calculate_polynomial_coefficients_for_tau_factor,
)
from napytau.core.tau import calculate_tau_i_values
from napytau.core.delta_tau import calculate_error_propagation_terms
from napytau.core.tau_final import calculate_tau_final
from typing import Tuple
import numpy as np
from napytau.import_export.model.dataset import DataSet


def calculate_lifetime_for_fit(
    dataset: DataSet, polynomial_degree: int
) -> Tuple[float, float]:
    """
    Docstring missing. To be implemented with issue #44.
    """
    # Fit the spline and get coefficients + knot sequence
    coefficients, knots = calculate_polynomial_coefficients_for_fit(
        dataset, polynomial_degree
    )

    # We now calculate the lifetimes tau_i for all measured distances
    tau_i_values: np.ndarray = calculate_tau_i_values(
        dataset,
        coefficients,
        knots,
        polynomial_degree,
    )

    # And we calculate the respective errors for the lifetimes
    delta_tau_i_values: np.ndarray = calculate_error_propagation_terms(
        dataset,
        coefficients,
        0,
        knots,
        polynomial_degree,
    )

    # From lifetimes and associated errors we can now calculate the weighted mean
    # and the uncertainty
    tau_final: Tuple[float, float] = calculate_tau_final(
        tau_i_values, delta_tau_i_values
    )

    return tau_final


def calculate_optimal_tau_factor(
    dataset: DataSet,
    t_hyp_range: Tuple[float, float],
    weight_factor: float,
    polynomial_degree: int,
) -> float:
    """
    Docstring missing. To be implemented with issue #44.
    """
    coefficients, knots = calculate_polynomial_coefficients_for_fit(
        dataset, polynomial_degree
    )

    optimal_t_hyp = optimize_tau_factor(
        dataset,
        weight_factor,
        coefficients,
        t_hyp_range,
        knots,
        polynomial_degree,
    )

    return optimal_t_hyp


def calculate_lifetime_for_custom_tau_factor(
    dataset: DataSet,
    custom_tau_factor: float,
    polynomial_degree: int,
) -> Tuple[float, float]:
    """
    Docstring missing. To be implemented with issue #44.
    """
    # Find spline coefficients such that P(t)/P'(t) = custom_tau_factor
    coefficients, knots = calculate_polynomial_coefficients_for_tau_factor(
        dataset,
        custom_tau_factor,
        polynomial_degree,
    )

    # We now calculate the lifetimes tau_i for all measured distances
    tau_i_values: np.ndarray = calculate_tau_i_values(
        dataset,
        coefficients,
        knots,
        polynomial_degree,
    )

    # And we calculate the respective errors for the lifetimes
    delta_tau_i_values: np.ndarray = calculate_error_propagation_terms(
        dataset,
        coefficients,
        custom_tau_factor,
        knots,
        polynomial_degree,
    )

    # From lifetimes and associated errors we can now calculate the weighted mean
    # and the uncertainty
    tau_final: Tuple[float, float] = calculate_tau_final(
        tau_i_values, delta_tau_i_values
    )

    return tau_final

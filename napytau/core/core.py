from typing import Tuple
from napytau.core.chi import optimize_tau_factor
from napytau.core.polynomials import (
    calculate_polynomial_coefficients_for_fit,
    calculate_polynomial_coefficients_for_tau_factor,
)
from napytau.core.tau import calculate_tau_i_values
from napytau.core.delta_tau import calculate_error_propagation_terms
from napytau.core.tau_final import calculate_tau_final
from napytau.import_export.model.dataset import DataSet
import numpy as np


def calculate_lifetime_for_fit(
    dataset: DataSet, polynomial_degree: int
) -> Tuple[float, float]:
    """
    Docstring missing. To be implemented with issue #44.
    """
    # Now we find the optimal coefficients for the given taufactor
    coefficients: np.ndarray = calculate_polynomial_coefficients_for_fit(
        dataset, polynomial_degree
    )

    # We now calculate the lifetimes tau_i for all measured distances
    tau_i_values: np.ndarray = calculate_tau_i_values(
        dataset,
        coefficients,
    )

    # And we calculate the respective errors for the lifetimes
    delta_tau_i_values: np.ndarray = calculate_error_propagation_terms(
        dataset,
        coefficients,
        0,
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
    coefficients: np.ndarray = calculate_polynomial_coefficients_for_fit(
        dataset, polynomial_degree
    )

    optimal_t_hyp = optimize_tau_factor(
        dataset,
        weight_factor,
        coefficients,
        t_hyp_range,
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
    # Now we find the optimal coefficients for the given taufactor
    coefficients: np.ndarray = calculate_polynomial_coefficients_for_tau_factor(
        dataset,
        custom_tau_factor,
        polynomial_degree,
    )

    # We now calculate the lifetimes tau_i for all measured distances
    tau_i_values: np.ndarray = calculate_tau_i_values(
        dataset,
        coefficients,
    )

    # And we calculate the respective errors for the lifetimes
    delta_tau_i_values: np.ndarray = calculate_error_propagation_terms(
        dataset,
        coefficients,
        custom_tau_factor,
    )

    # From lifetimes and associated errors we can now calculate the weighted mean
    # and the uncertainty
    tau_final: Tuple[float, float] = calculate_tau_final(
        tau_i_values, delta_tau_i_values
    )

    return tau_final

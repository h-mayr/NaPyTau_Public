from napytau.core.polynomials import (
    calculate_polynomial_coefficients_for_coupled_fit,
    calculate_polynomial_coefficients_for_fit,
    evaluate_differentiated_polynomial_at_measuring_times,
    evaluate_polynomial_at_measuring_times,
)
import numpy as np
from numpy.random import Generator

from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.dataset import DataSet
from napytau.util.model.value_error_pair import ValueErrorPair


def calculate_jacobian_matrix(
    dataset: DataSet,
    coefficients: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Calculates the Jacobian matrix for a set of B-spline coefficients by
    numerically perturbing each coefficient.

    Args:
        dataset (DataSet): The dataset of the experiment
        coefficients (ndarray): B-spline coefficients
        knots (ndarray): Full B-spline knot sequence
        degree (int): Degree of the B-spline

    Returns:
        ndarray:
        The computed Jacobian matrix with shape (len(distances), len(coefficients)).
    """

    datapoints = dataset.get_datapoints()
    # initializes the jacobian matrix
    jacobian_matrix: np.ndarray = np.zeros(
        (len(datapoints.get_distances().get_values()), len(coefficients))
    )

    epsilon: float = 1e-6  # small disturbance value

    # Loop over each coefficient and calculate the partial derivative
    for i in range(len(coefficients)):
        perturbed_coefficients: np.ndarray = np.array(coefficients, dtype=float)
        perturbed_coefficients[i] += epsilon  # slightly disturb the current coefficient

        # Compute the disturbed and original spline values at the given distances
        perturbed_function: np.ndarray = evaluate_polynomial_at_measuring_times(
            dataset, perturbed_coefficients, knots, degree
        )
        original_function: np.ndarray = evaluate_polynomial_at_measuring_times(
            dataset, coefficients, knots, degree
        )

        # Calculate the partial derivative and store it in the Jacobian matrix
        jacobian_matrix[:, i] = (perturbed_function - original_function) / epsilon

    return jacobian_matrix


def calculate_covariance_matrix(
    dataset: DataSet,
    coefficients: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Computes the covariance matrix for the B-spline coefficients using the
    Jacobian matrix and a weight matrix derived from the shifted intensities' errors.

    Args:
        dataset (DataSet): The dataset of the experiment
        coefficients (ndarray): B-spline coefficients
        knots (ndarray): Full B-spline knot sequence
        degree (int): Degree of the B-spline

    Returns:
        ndarray: The computed covariance matrix for the B-spline coefficients.
    """

    datapoints = dataset.get_datapoints()
    jacobian_matrix: np.ndarray = calculate_jacobian_matrix(
        dataset, coefficients, knots, degree
    )

    # Construct the weight matrix from the inverse squared errors
    weight_matrix: np.ndarray = np.diag(
        1 / np.power(datapoints.get_shifted_intensities().get_errors(), 2)
    )

    fit_matrix: np.ndarray = jacobian_matrix.T @ weight_matrix @ jacobian_matrix

    covariance_matrix: np.ndarray = np.linalg.pinv(fit_matrix)

    return covariance_matrix


def calculate_error_propagation_terms(
    dataset: DataSet,
    coefficients: np.ndarray,
    taufactor: float,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Creates the error propagation terms for the B-spline coefficients,
    combining direct errors, spline uncertainties, and mixed covariance terms.

    Args:
        dataset (DataSet): The dataset of the experiment
        coefficients (ndarray): B-spline coefficients
        taufactor (float): Scaling factor related to the Doppler-shift model.
        knots (ndarray): Full B-spline knot sequence
        degree (int): Degree of the B-spline

    Returns:
        ndarray: The combined error propagation terms for each distance point.
    """

    datapoints = dataset.get_datapoints()
    calculated_differentiated_polynomial_sum_at_measuring_distances = (
        evaluate_differentiated_polynomial_at_measuring_times(
            dataset,
            coefficients,
            knots,
            degree,
        )
    )

    gaussian_error_from_unshifted_intensity: np.ndarray = np.power(
        datapoints.get_unshifted_intensities().get_errors(), 2
    ) / np.power(
        calculated_differentiated_polynomial_sum_at_measuring_distances,
        2,
    )

    jacobian_matrix: np.ndarray = calculate_jacobian_matrix(
        dataset, coefficients, knots, degree
    )
    covariance_matrix: np.ndarray = calculate_covariance_matrix(
        dataset, coefficients, knots, degree
    )

    # Var[P(t_i)] = J[i,:] @ Cov @ J[i,:]^T — correct for any basis
    delta_p_j_i_squared: np.ndarray = np.einsum(
        "ij,jk,ik->i", jacobian_matrix, covariance_matrix, jacobian_matrix
    )

    gaussian_error_from_polynomial_uncertainties: np.ndarray = (
        np.power(datapoints.get_unshifted_intensities().get_values(), 2)
        / np.power(
            calculated_differentiated_polynomial_sum_at_measuring_distances,
            4,
        )
    ) * np.power(delta_p_j_i_squared, 2)

    error_from_covariance: np.ndarray = (
        datapoints.get_unshifted_intensities().get_values()
        * taufactor
        * delta_p_j_i_squared
    ) / np.power(calculated_differentiated_polynomial_sum_at_measuring_distances, 3)

    interim_result: np.ndarray = (
        gaussian_error_from_unshifted_intensity
        + gaussian_error_from_polynomial_uncertainties
    )
    errors: np.ndarray = interim_result + error_from_covariance
    # Return the sum of all three contributions
    return errors


def _perturb_dataset(dataset: DataSet, rng: Generator) -> DataSet:
    """Return a copy of dataset with intensities perturbed by N(0, sigma)."""
    perturbed_points: list[Datapoint] = []
    for dp in dataset.get_datapoints():
        sh, us = dp.get_intensity()
        new_sh = rng.normal(sh.value, abs(sh.error))
        new_us = rng.normal(us.value, abs(us.error))
        new_dp = Datapoint(
            distance=dp.distance,
            calibration=dp.calibration,
            shifted_intensity=ValueErrorPair(float(new_sh), sh.error),
            unshifted_intensity=ValueErrorPair(float(new_us), us.error),
            feeding_shifted_intensity=dp.feeding_shifted_intensity,
            feeding_unshifted_intensity=dp.feeding_unshifted_intensity,
            active=dp.active,
        )
        perturbed_points.append(new_dp)
    return DataSet(
        relative_velocity=dataset.get_relative_velocity(),
        datapoints=DatapointCollection(perturbed_points),
        sampling_points=dataset.get_sampling_points(),
    )


def calculate_error_propagation_mc(
    dataset: DataSet,
    degree: int,
    smoothing_factor: float | None,
    fit_mode: str,
    n_iterations: int = 100,
) -> tuple[float, float]:
    """
    Returns (tau_final, sigma_tau) estimated via Monte Carlo resampling.

    For each iteration:
      1. Perturb I_sh[i] ~ N(I_sh[i], σ_sh[i]) and I_us[i] ~ N(I_us[i], σ_us[i])
      2. Refit and compute weighted-mean τ_final
    Final sigma = std of the τ_final distribution.

    Args:
        dataset (DataSet): The dataset of the experiment
        degree (int): B-spline degree
        smoothing_factor (float | None): Smoothing factor (None = LSQ mode)
        fit_mode (str): "lsq", "smooth", or "coupled"
        n_iterations (int): Number of Monte Carlo samples

    Returns:
        tuple[float, float]: (mean tau_final, std of tau_final distribution)
    """
    from napytau.core.tau import calculate_tau_i_values
    from napytau.core.tau_final import calculate_tau_final

    rng = np.random.default_rng()
    tau_samples: list[float] = []

    for _ in range(n_iterations):
        perturbed = _perturb_dataset(dataset, rng)
        try:
            if fit_mode == "coupled":
                coefficients, knots = calculate_polynomial_coefficients_for_coupled_fit(
                    perturbed, 1.0, degree
                )
            else:
                coefficients, knots = calculate_polynomial_coefficients_for_fit(
                    perturbed, degree, smoothing_factor
                )
            tau_i = calculate_tau_i_values(perturbed, coefficients, knots, degree)
            delta_tau_i = calculate_error_propagation_terms(
                perturbed, coefficients, 0.0, knots, degree
            )
            tau_mc, _ = calculate_tau_final(tau_i, delta_tau_i)
            if np.isfinite(tau_mc) and tau_mc != -1:
                tau_samples.append(tau_mc)
        except Exception:
            continue

    if len(tau_samples) == 0:
        return -1.0, -1.0

    arr = np.array(tau_samples)
    return float(np.mean(arr)), float(np.std(arr))

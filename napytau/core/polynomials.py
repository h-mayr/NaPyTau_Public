import numpy as np
import scipy as sp

from napytau.core.errors.polynomial_coefficient_error import (
    PolynomialCoefficientError,
)
from napytau.core.time import (
    calculate_time_from_distance_and_relative_velocity,
    calculate_times_from_distances_and_relative_velocity,
)
from napytau.import_export.model.dataset import DataSet


def evaluate_polynomial_at_measuring_times(
    dataset: DataSet,
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Computes the sum of a polynomial evaluated at given time points.

    Args:
        dataset (DataSet): The dataset of the experiment
        Datapoints for fitting, consisting of distances and intensities
        coefficients (ndarray):
        Array of polynomial coefficients [a_0, a_1, ..., a_n],
        where the polynomial is P(t) = a_0 + a_1*t + a_2*t^2 + ... + a_n*t^n.

    Returns:
        ndarray: Array of polynomial values evaluated at the given time points.
    """
    if len(coefficients) == 0:
        raise PolynomialCoefficientError(
            "An empty array of coefficients can not be evaluated."
        )

    times: np.ndarray = calculate_times_from_distances_and_relative_velocity(dataset)
    # Evaluate the polynomial sum at the given time points
    sum_at_measuring_distances: np.ndarray = np.zeros_like(times, dtype=float)
    for exponent, coefficient in enumerate(coefficients):
        sum_at_measuring_distances += coefficient * np.power(times, exponent)

    return sum_at_measuring_distances


def evaluate_polynomial_at_measuring_time(
    dataset: DataSet,
    distance: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Computes the sum of a polynomial evaluated at given time points.

    Args:
        dataset (DataSet): The dataset of the experiment
        distance (float): Distance at which the polynomial is evaluated
        Datapoints for fitting, consisting of distances and intensities
        coefficients (ndarray):
        Array of polynomial coefficients [a_0, a_1, ..., a_n],
        where the polynomial is P(t) = a_0 + a_1*t + a_2*t^2 + ... + a_n*t^n.

    Returns:
        ndarray: Array of polynomial values evaluated at the given time points.
    """

    if len(coefficients) == 0:
        raise PolynomialCoefficientError(
            "An empty array of coefficients can not be evaluated."
        )

    time: float = calculate_time_from_distance_and_relative_velocity(dataset, distance)
    # Evaluate the polynomial sum at the given time points
    sum_at_measuring_distance: float = 0.0
    for exponent, coefficient in enumerate(coefficients):
        sum_at_measuring_distance += coefficient * pow(time, exponent)

    return sum_at_measuring_distance


def evaluate_differentiated_polynomial_at_measuring_times(
    dataset: DataSet,
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Computes the sum of the derivative of a polynomial evaluated
    at given time points.

    Args:
        dataset (DataSet): The dataset of the experiment
        Datapoints for fitting, consisting of distances and intensities
        coefficients (ndarray):
        Array of polynomial coefficients [a_0, a_1, ..., a_n],
        where the polynomial is P(t) = a_0 + a_1*t + a_2*t^2 + ... + a_n*t^n.

    Returns:
        ndarray:
        Array of the derivative values of the polynomial at the given time points.
    """
    if len(coefficients) == 0:
        raise PolynomialCoefficientError(
            "An empty array of coefficients can not be evaluated."
        )

    times: np.ndarray = calculate_times_from_distances_and_relative_velocity(dataset)
    sum_of_derivative_at_measuring_distances: np.ndarray = np.zeros_like(
        times, dtype=float
    )
    for exponent, coefficient in enumerate(coefficients):
        if exponent > 0:
            sum_of_derivative_at_measuring_distances += (
                exponent * coefficient * np.power(times, (exponent - 1))
            )

    return sum_of_derivative_at_measuring_distances


def calculate_polynomial_coefficients_for_fit(
    dataset: DataSet,
    degree: int,
) -> np.ndarray:
    """
    Calculates the polynomial coefficients for the polynomial fit.

    Args:
        dataset (DataSet): The dataset of the experiment
        degree (int): The degree of the polynomial to be fitted

    Returns:
        ndarray: Array of polynomial coefficients for the fit.
    """
    # Calculate the polynomial coefficients for the fit
    polynomial_coefficients: np.ndarray = (
        np.polynomial.Polynomial.fit(
            calculate_times_from_distances_and_relative_velocity(dataset),
            dataset.get_datapoints().get_unshifted_intensities().get_values(),
            degree - 1,
        )
        .convert()
        .coef
    )

    return polynomial_coefficients


def calculate_polynomial_coefficients_for_tau_factor(
    dataset: DataSet,
    tau_factor: float,
    degree: int,
) -> np.ndarray:
    """
    Calculates the polynomial coefficients for the tau factor.

    Args:
        dataset (DataSet): The dataset of the experiment
        tau_factor (float): The tau factor to be used in the polynomial fit
        degree (int): The degree of the polynomial to be fitted

    Returns:
        ndarray: Array of polynomial coefficients for the tau factor.
    """

    def polynomial_fit(x, *coefficients):
        # ensure that there is no zero in the coefficients to avoid division by zero
        if 0 in coefficients:
            for i in range(len(coefficients)):
                if coefficients[i] == 0:
                    coefficients = tuple(
                        list(coefficients[:i]) + [1e-10] + list(coefficients[i + 1 :])
                    )
        return (
            np.poly1d(coefficients)(x) / np.polyder(np.poly1d(coefficients))(x)
        ) - tau_factor

    # Initial guess: coefficients as ones
    initial_guess = np.ones(degree + 1)

    # Solve for coefficients using least squares
    res = sp.optimize.least_squares(
        lambda coefficients: polynomial_fit(
            calculate_times_from_distances_and_relative_velocity(dataset),
            *coefficients,
        ),
        initial_guess,
    )

    return np.array(res.x)

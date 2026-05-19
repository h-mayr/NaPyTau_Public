from napytau.core.polynomials import (
    evaluate_differentiated_polynomial_at_measuring_times,
)  # noqa E501
import numpy as np
from napytau.import_export.model.dataset import DataSet


def calculate_tau_i_values(
    dataset: DataSet,
    coefficients: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Calculates the decay times (tau_i) based on the provided
    intensities and time points.

    Args:
        dataset (DataSet): The dataset of the experiment
        coefficients (ndarray): B-spline coefficients
        knots (ndarray): Full B-spline knot sequence
        degree (int): Degree of the B-spline

    Returns:
        ndarray: Calculated decay times for each distance point.
    """

    # calculate decay times using the optimized coefficients
    tau_i_values: np.ndarray = (
        dataset.get_datapoints().get_normalized_unshifted_intensities().get_values()
        / evaluate_differentiated_polynomial_at_measuring_times(
            dataset, coefficients, knots, degree
        )
    )

    return tau_i_values

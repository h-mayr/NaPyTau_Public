import numpy as np
from napytau.core.polynomials import (
    evaluate_differentiated_polynomial_at_measuring_times,
)
from napytau.import_export.model.dataset import DataSet


def calculate_tau_i_values(
    dataset: DataSet,
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Calculates the decay times (tau_i) based on the provided
    intensities and time points.

    Args:
        dataset (DataSet): The dataset of the experiment
        initial_coefficients (ndarray):
        Initial guess for the polynomial coefficients
        t_hyp_range (tuple):
        Range for hypothesis optimization (min, max)
        weight_factor (float):
        Weighting factor for unshifted intensities

    Returns:
        ndarray: Calculated decay times for each distance point.
    """

    # calculate decay times using the optimized coefficients
    tau_i_values: (
        np.ndarray
    ) = dataset.get_datapoints().get_unshifted_intensities().get_values() / evaluate_differentiated_polynomial_at_measuring_times(
        dataset, coefficients
    )

    return tau_i_values

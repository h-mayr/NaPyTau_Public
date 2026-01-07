import numpy as np
from typing import Tuple

from napytau.import_export.model.dataset import DataSet
from napytau.core.time import calculate_times_from_distances_and_relative_velocity


def _prepare_intensity_arrays(dataset):
    """Return the normalised peak areas and their uncertainties as intensity arrays."""
    normalisation_factors = dataset.get_datapoints().get_calibrations().get_values()
    normalisation_factors_uncertainties = (
        dataset.get_datapoints().get_calibrations().get_errors()
    )
    intensities_unshifted = (
        dataset.get_datapoints().get_unshifted_intensities().get_values()
        * normalisation_factors
    )
    intensities_unshifted_uncertainties = np.sqrt(
        (
            dataset.get_datapoints().get_unshifted_intensities().get_errors()
            * normalisation_factors
        )
        ** 2
        + (
            normalisation_factors_uncertainties
            * dataset.get_datapoints().get_unshifted_intensities().get_values()
        )
        ** 2
    )
    intensities_shifted = (
        dataset.get_datapoints().get_shifted_intensities().get_values()
        * normalisation_factors
    )
    intensities_shifted_uncertainties = np.sqrt(
        (
            dataset.get_datapoints().get_shifted_intensities().get_errors()
            * normalisation_factors
        )
        ** 2
        + (
            dataset.get_datapoints().get_shifted_intensities().get_values()
            * normalisation_factors_uncertainties
        )
        ** 2
    )
    flight_times = calculate_times_from_distances_and_relative_velocity(dataset)

    return (
        intensities_unshifted,
        intensities_unshifted_uncertainties,
        intensities_shifted,
        intensities_shifted_uncertainties,
        flight_times,
    )


def calc_tau_simple(dataset) -> Tuple[float, float]:
    """
    Calculate tau_simple the same way as napatau does using lambda.

    This reproduces results of napatau by the "calc_tau_simple" function or clicking
    the "tau simple button" in its GUI.
    The decay constants lambda are calculated for each distance. Then the weighted mean
    is calculated and afterwards inverted to get the final mean value for the lifetime.
    """
    (
        intensities_unshifted,
        intensities_unshifted_uncertainties,
        intensities_shifted,
        intensities_shifted_uncertainties,
        flight_times,
    ) = _prepare_intensity_arrays(dataset)

    n_pairs = len(intensities_unshifted) - 1
    unshifted_mean = np.empty(n_pairs)
    unshifted_mean_uncertainties = np.empty(n_pairs)
    differential_shifted_mean = np.empty(n_pairs)
    differential_shifted_mean_uncertainties = np.empty(n_pairs)
    lambda_decay_constant = np.empty(n_pairs)
    lambda_decay_constant_uncertainty = np.empty(n_pairs)

    for i in range(n_pairs):
        unshifted_mean[i] = 0.5 * (
            intensities_unshifted[i + 1] + intensities_unshifted[i]
        )
        unshifted_mean_uncertainties[i] = np.sqrt(
            0.25
            * (
                intensities_unshifted_uncertainties[i + 1] ** 2
                + intensities_unshifted_uncertainties[i] ** 2
            )
        )

        dt = flight_times[i + 1] - flight_times[i]
        differential_shifted_mean[i] = (
            intensities_shifted[i + 1] - intensities_shifted[i]
        ) / dt
        differential_shifted_mean_uncertainties[i] = np.sqrt(
            (1.0 / dt) ** 2
            * (
                intensities_shifted_uncertainties[i + 1] ** 2
                + intensities_shifted_uncertainties[i] ** 2
            )
        )

        lambda_decay_constant[i] = differential_shifted_mean[i] / unshifted_mean[i]
        lambda_decay_constant_uncertainty[i] = np.sqrt(
            (differential_shifted_mean_uncertainties[i] / unshifted_mean[i]) ** 2
            + (
                differential_shifted_mean[i]
                * unshifted_mean_uncertainties[i]
                / (unshifted_mean[i] ** 2)
            )
            ** 2
        )

    weights = 1.0 / (lambda_decay_constant_uncertainty**2)
    weights_sum = np.sum(weights)
    lambda_mean = np.sum(weights * lambda_decay_constant) / weights_sum
    lambda_mean_err = 1.0 / np.sqrt(weights_sum)

    tau_simple = 1.0 / lambda_mean
    tau_simple_uncertainty = (1.0 / lambda_mean**2) * lambda_mean_err

    return tau_simple, tau_simple_uncertainty


def calc_tau_simple_alternative(dataset: DataSet) -> Tuple[float, float]:
    """
    Alternative tau_simple calculation.

    Calculate tau_simple directly from tau and not via lambda.
    This gives significantly different results since 1/x is non-linear:
        mean[τ] = mean[1/λ] ≠ 1/mean[λ]
    Napatau calculates first the mean of all λ_i and then averages it. This is different
    as calculating the average of τ_i. Which approach is more physical is out of scope
    of this implementation. Discussions regarding this topic are welcome.
    """
    (
        intensities_unshifted,
        intensities_unshifted_uncertainties,
        intensities_shifted,
        intensities_shifted_uncertainties,
        flight_times,
    ) = _prepare_intensity_arrays(dataset)

    unshifted_mean = np.empty((len(intensities_unshifted) - 1))
    unshifted_mean_uncertainties = np.empty((len(intensities_unshifted) - 1))
    differential_shifted_mean = np.empty((len(intensities_shifted) - 1))
    differential_shifted_mean_uncertainties = np.empty((len(intensities_shifted) - 1))
    for i in range(len(intensities_unshifted) - 1):
        unshifted_mean[i] = 0.5 * (
            intensities_unshifted[i + 1] + intensities_unshifted[i]
        )
        unshifted_mean_uncertainties[i] = np.sqrt(
            0.25
            * (
                intensities_unshifted_uncertainties[i + 1] ** 2
                + intensities_unshifted_uncertainties[i] ** 2
            )
        )

        differential_shifted_mean[i] = (
            intensities_shifted[i + 1] - intensities_shifted[i]
        ) / (flight_times[i + 1] - flight_times[i])

        differential_shifted_mean_uncertainties[i] = np.sqrt(
            (1 / (flight_times[i + 1] - flight_times[i])) ** 2
            * (
                intensities_shifted_uncertainties[i + 1] ** 2
                + intensities_shifted_uncertainties[i] ** 2
            )
        )

    tau_simple_values = unshifted_mean / differential_shifted_mean
    tau_simple_uncertainties = np.sqrt(
        (unshifted_mean_uncertainties / differential_shifted_mean) ** 2
        + (
            differential_shifted_mean_uncertainties
            * unshifted_mean
            / differential_shifted_mean**2
        )
        ** 2
    )
    tau_simple, sum_of_weights = np.average(
        tau_simple_values, weights=1 / tau_simple_uncertainties**2, returned=True
    )
    tau_simple_uncertainty = 1 / np.sqrt(sum_of_weights)

    return tau_simple, tau_simple_uncertainty

import numpy as np

from napytau.import_export.model.dataset import DataSet
from napytau.core.time import calculate_times_from_distances_and_relative_velocity


def calc_tau_simple(dataset: DataSet) -> np.ndarray:
    normalisation_factors = dataset.get_datapoints().get_calibrations().get_values()
    intensities_unshifted = (
        dataset.get_datapoints().get_unshifted_intensities().get_values()
        * normalisation_factors
    )
    intensities_unshifted_uncertainties = (
        dataset.get_datapoints().get_unshifted_intensities().get_errors()
        * normalisation_factors
    )
    intensities_shifted = (
        dataset.get_datapoints().get_shifted_intensities().get_values()
        * normalisation_factors
    )
    intensities_shifted_uncertainties = (
        dataset.get_datapoints().get_shifted_intensities().get_errors()
        * normalisation_factors
    )
    distances = dataset.get_datapoints().get_distances().get_values()
    flight_times = calculate_times_from_distances_and_relative_velocity(dataset)

    unshifted_mean_uncertainties = np.empty((len(intensities_unshifted) - 1))
    unshifted_mean = np.empty((len(intensities_unshifted) - 1))
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
            (1 / (flight_times[i + 1] - flight_times[i]) ** 2)
            * (intensities_shifted[i + 1] ** 2 + intensities_shifted[i] ** 2)
        )

    tau_simple_values = unshifted_mean / differential_shifted_mean
    tau_simple_uncertainties = np.sqrt(
        (unshifted_mean_uncertainties / differential_shifted_mean) ** 2
        + (differential_shifted_mean * unshifted_mean / differential_shifted_mean**2)
        ** 2
    )
    tau_simple = np.average(tau_simple_values, weights=1 / tau_simple_uncertainties**2)
    tau_simple_uncertainty = np.sqrt(np.sum(tau_simple_uncertainties**2)) / len(
        tau_simple_uncertainties
    )

    return tau_simple, tau_simple_uncertainty

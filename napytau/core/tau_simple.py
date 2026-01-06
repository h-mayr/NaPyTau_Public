import numpy as np
from napytau.import_export.model.dataset import DataSet
from napytau.core.time import calculate_times_from_distances_and_relative_velocity


def calc_tau_simple(dataset: DataSet) -> np.ndarray:
    intensities_unshifted = (
        dataset.get_datapoints().get_unshifted_intensities().get_values()
    )
    intensities_shifted = (
        dataset.get_datapoints().get_shifted_intensities().get_values()
    )
    distances = dataset.get_datapoints().get_distances().get_values()
    flight_times = calculate_times_from_distances_and_relative_velocity(dataset)

    unshifted_mean = np.empty((len(intensities_unshifted) - 1))
    differential_shifted_mean = np.empty((len(intensities_shifted) - 1))
    for i in range(len(intensities_unshifted) - 1):
        unshifted_mean[i] = 0.5 * (
            intensities_unshifted[i + 1] + intensities_unshifted[i]
        )

        differential_shifted_mean[i] = (
            intensities_shifted[i + 1] - intensities_unshifted[i]
        ) / (flight_times[i + 1] - flight_times[i])

    tau_simple_values = unshifted_mean / differential_shifted_mean
    tau_simple = np.mean(tau_simple_values)

    return tau_simple

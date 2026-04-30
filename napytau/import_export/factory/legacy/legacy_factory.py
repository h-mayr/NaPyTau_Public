from typing import List, Optional, Tuple

from napytau.import_export.factory.legacy.raw_legacy_data import RawLegacyData

from napytau.import_export.factory.legacy.raw_legacy_setup_data import (
    RawLegacySetupData,
)
from napytau.import_export.import_export_error import ImportExportError
from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity
from napytau.util.model.ValueErrorPairCollection import ValueErrorPairCollection
from napytau.util.model.value_error_pair import ValueErrorPair


class LegacyFactory:
    """
    A factory class for creating datasets from raw data stored in the Legacy format.
    """

    @staticmethod
    def create_dataset(raw_dataset: RawLegacyData) -> DataSet:
        return DataSet(
            LegacyFactory.parse_velocity(raw_dataset.velocity_rows),
            LegacyFactory.parse_datapoints(
                raw_dataset.distance_rows,
                raw_dataset.normalisation_rows,
                raw_dataset.fit_rows,
            ),
        )

    @staticmethod
    def parse_velocity(velocity_rows: List[str]) -> ValueErrorPair[RelativeVelocity]:
        filtered_velocities = list(
            filter(lambda x: not x.startswith("#"), velocity_rows)
        )

        if len(filtered_velocities) != 1:
            raise ValueError(
                f"Expected one velocity row, but got {len(filtered_velocities)}"
            )

        split_velocity_row = filtered_velocities[0].split(" ")

        if len(split_velocity_row) < 1:
            raise ValueError(
                f"Expected at least 1 value in velocity row, but got {len(split_velocity_row)}"  # noqa E501
            )

        if len(split_velocity_row) > 2:
            raise ValueError(
                f"Expected at most 1 value in velocity row, but got {len(split_velocity_row)}"  # noqa E501
            )

        velocity = float(split_velocity_row[0])
        velocity_error = (
            float(split_velocity_row[1]) if len(split_velocity_row) == 2 else 0.0
        )

        return ValueErrorPair(
            RelativeVelocity(velocity), RelativeVelocity(velocity_error)
        )

    @staticmethod
    def parse_datapoints(
        distance_rows: List[str],
        normalisation_rows: List[str],
        fit_rows: List[str],
    ) -> DatapointCollection:
        datapoints = DatapointCollection([])
        for distance_row in distance_rows:
            distance = LegacyFactory.parse_distance_row(distance_row)
            datapoints.add_datapoint(Datapoint(distance))
        for normalisation_row in normalisation_rows:
            (distance_index, normalisation) = LegacyFactory.parse_normalisation_row(
                normalisation_row
            )
            datapoint = datapoints.get_datapoint_by_distance(distance_index)
            datapoint.set_normalisation(normalisation)
        for fit_row in fit_rows:
            (
                distance_index,
                shifted_intensity,
                unshifted_intensity,
                feeding_shifted_intensity,
                feeding_unshifted_intensity,
            ) = LegacyFactory.parse_fit_row(fit_row)
            datapoint = datapoints.get_datapoint_by_distance(distance_index)
            datapoint.set_intensity(shifted_intensity, unshifted_intensity)
            if (
                feeding_shifted_intensity is not None
                and feeding_unshifted_intensity is not None
            ):
                datapoint.set_feeding_intensity(
                    feeding_shifted_intensity, feeding_unshifted_intensity
                )

        return datapoints

    @staticmethod
    def parse_distance_row(distance_row: str) -> ValueErrorPair[float]:
        split_row = distance_row.split(" ")
        if len(split_row) < 3:
            raise ValueError(
                f"Expected at least 3 values in distance row, but got {len(split_row)}"
            )

        if len(split_row) > 3:
            raise ValueError(
                f"Expected at most 3 values in distance row, but got {len(split_row)}"
            )

        # The first value (at index 0) is a label, however since we index by distance we can ignore it # noqa E501
        distance = float(split_row[1])
        distance_error = float(split_row[2])

        return ValueErrorPair(distance, distance_error)

    @staticmethod
    def parse_normalisation_row(
        normalisation_row: str,
    ) -> Tuple[float, ValueErrorPair[float]]:
        split_row = normalisation_row.split(" ")
        if len(split_row) < 3:
            raise ValueError(
                f"Expected at least 3 values in normalisation row, but got {len(split_row)}"  # noqa E501
            )

        if len(split_row) > 3:
            raise ValueError(
                f"Expected at most 3 values in normalisation row, but got {len(split_row)}"  # noqa E501
            )

        distance_index = float(split_row[0])
        normalisation = float(split_row[1])
        normalisation_error = float(split_row[2])

        return distance_index, ValueErrorPair(normalisation, normalisation_error)

    @staticmethod
    def parse_fit_row(
        fit_row: str,
    ) -> Tuple[
        float,
        ValueErrorPair[float],
        ValueErrorPair[float],
        Optional[ValueErrorPair[float]],
        Optional[ValueErrorPair[float]],
    ]:
        split_row = fit_row.split(" ")
        if len(split_row) < 5:
            raise ValueError(
                f"Expected at least 5 values in fit row, but got {len(split_row)}"
            )

        distance_index = float(split_row[0])
        shifted_intensity = float(split_row[1])
        shifted_intensity_error = float(split_row[2])
        unshifted_intensity = float(split_row[3])
        unshifted_intensity_error = float(split_row[4])

        if len(split_row) == 5:
            return (
                distance_index,
                ValueErrorPair(shifted_intensity, shifted_intensity_error),
                ValueErrorPair(unshifted_intensity, unshifted_intensity_error),
                None,
                None,
            )

        if len(split_row) < 9:
            raise ValueError(
                f"Expected at least 9 values in fit row, but got {len(split_row)}"
            )

        if len(split_row) > 9:
            raise ValueError(
                f"Expected at most 9 values in fit row, but got {len(split_row)}"
            )

        feeding_shifted_intensity = float(split_row[5])
        feeding_shifted_intensity_error = float(split_row[6])
        feeding_unshifted_intensity = float(split_row[7])
        feeding_unshifted_intensity_error = float(split_row[8])

        return (
            distance_index,
            ValueErrorPair(shifted_intensity, shifted_intensity_error),
            ValueErrorPair(unshifted_intensity, unshifted_intensity_error),
            ValueErrorPair(feeding_shifted_intensity, feeding_shifted_intensity_error),
            ValueErrorPair(
                feeding_unshifted_intensity, feeding_unshifted_intensity_error
            ),
        )

    @staticmethod
    def enrich_dataset(dataset: DataSet, raw_setup_data: RawLegacySetupData) -> DataSet:
        try:
            datapoint_count = len(dataset.datapoints)
            tau_row = raw_setup_data.napsetup_rows[0]
            datapoint_active_rows = raw_setup_data.napsetup_rows[
                1 : (datapoint_count + 1)
            ]
            polynomial_count_row = raw_setup_data.napsetup_rows[datapoint_count + 1]
            sampling_points_row = raw_setup_data.napsetup_rows[
                datapoint_count + 2 : len(raw_setup_data.napsetup_rows)
            ]
        except IndexError as e:
            raise ImportExportError(
                "The provided Legacy setup file is not formatted correctly. Please check the file."  # noqa E501
            ) from e

        try:
            dataset.set_tau_factor(LegacyFactory.parse_tau_factor(tau_row))
        except ValueError as e:
            raise ImportExportError(
                "The tau factor provided in the Legacy setup file is not formatted correctly. Please check the file."  # noqa E501
            ) from e

        try:
            for distance, active in LegacyFactory.parse_datapoint_active_rows(
                datapoint_active_rows,
                dataset.get_datapoints().get_distances(),
            ):
                dataset.datapoints.get_datapoint_by_distance(distance).set_active(
                    active
                )
        except ValueError as e:
            raise ImportExportError(
                "The active rows provided in the Legacy setup file are not formatted correctly. Please check the file."  # noqa E501
            ) from e

        try:
            dataset.set_polynomial_count(
                LegacyFactory.parse_polynomial_count(polynomial_count_row)
            )
        except ValueError as e:
            raise ImportExportError(
                "The polynomial count provided in the Legacy setup file is not formatted correctly. Please check the file."  # noqa E501
            ) from e

        try:
            dataset.set_sampling_points(
                LegacyFactory.parse_sampling_points(sampling_points_row)
            )
        except ValueError as e:
            raise ImportExportError(
                "The sampling points provided in the Legacy setup file are not formatted correctly. Please check the file."  # noqa E501
            ) from e

        return dataset

    @staticmethod
    def parse_tau_factor(tau_row: str) -> float:
        split_row = tau_row.split()

        return float(split_row[0])

    @staticmethod
    def parse_datapoint_active_rows(
        active_rows: List[str],
        distances: ValueErrorPairCollection[float],
    ) -> List[Tuple[float, bool]]:
        active_datapoints = []
        for index, active_row in enumerate(active_rows):
            split_row = active_row.split()
            active = bool(int(split_row[0]))
            distance = distances[index].value

            active_datapoints.append((distance, active))

        return active_datapoints

    @staticmethod
    def parse_polynomial_count(polynomial_count_row: str) -> int:
        split_row = polynomial_count_row.split()

        return int(split_row[0])

    @staticmethod
    def parse_sampling_points(sampling_points_row: List[str]) -> List[float]:
        sampling_points = []
        for sampling_point_row in sampling_points_row:
            split_row = sampling_point_row.split()
            sampling_points.append(float(split_row[0]))

        return sampling_points

from typing import List

from napytau.import_export.factory.napytau.json_service.napytau_format_json_service import (  # noqa E501
    NapytauFormatJsonService,
)
from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity
from napytau.util.coalesce import coalesce
from napytau.util.model.value_error_pair import ValueErrorPair


class NapyTauFactory:
    @staticmethod
    def create_dataset(raw_json_data: dict) -> DataSet:
        NapytauFormatJsonService.validate_against_schema(raw_json_data)

        return DataSet(
            ValueErrorPair(
                RelativeVelocity(raw_json_data["relativeVelocity"]),
                RelativeVelocity(raw_json_data["relativeVelocityError"]),
            ),
            NapyTauFactory._parse_datapoints(raw_json_data["datapoints"]),
        )

    @staticmethod
    def _parse_datapoints(
        raw_datapoints: List[dict[str, float]],
    ) -> DatapointCollection:
        datapoints = []
        for raw_datapoint in raw_datapoints:
            distance = ValueErrorPair(
                coalesce(raw_datapoint.get("distance")),
                coalesce(raw_datapoint.get("distanceError")),
            )
            normalisation = ValueErrorPair(
                coalesce(raw_datapoint.get("normalisation")),
                coalesce(raw_datapoint.get("normalisationError")),
            )
            shifted_intensity = ValueErrorPair(
                coalesce(raw_datapoint.get("shiftedIntensity")),
                coalesce(raw_datapoint.get("shiftedIntensityError")),
            )
            unshifted_intensity = ValueErrorPair(
                coalesce(raw_datapoint.get("unshiftedIntensity")),
                coalesce(raw_datapoint.get("unshiftedIntensityError")),
            )

            if "feedingShiftedIntensity" in raw_datapoint:
                feeding_shifted_intensity = ValueErrorPair(
                    coalesce(raw_datapoint.get("feedingShiftedIntensity")),
                    coalesce(raw_datapoint.get("feedingShiftedIntensityError")),
                )
                feeding_unshifted_intensity = ValueErrorPair(
                    coalesce(raw_datapoint.get("feedingUnshiftedIntensity")),
                    coalesce(raw_datapoint.get("feedingUnshiftedIntensityError")),
                )
            else:
                feeding_shifted_intensity = None
                feeding_unshifted_intensity = None

            datapoints.append(
                Datapoint(
                    distance,
                    normalisation,
                    shifted_intensity,
                    unshifted_intensity,
                    feeding_shifted_intensity,
                    feeding_unshifted_intensity,
                )
            )

        return DatapointCollection(datapoints)

    @staticmethod
    def enrich_dataset(dataset: DataSet, setup: dict) -> DataSet:
        dataset.set_tau_factor(setup["tauFactor"])

        dataset.set_polynomial_count(setup["polynomialCount"])

        for datapoint_setup in setup["datapointSetups"]:
            datapoint = dataset.get_datapoints().get_datapoint_by_distance(
                datapoint_setup["distance"]
            )

            datapoint.set_active(datapoint_setup["active"])

        return dataset

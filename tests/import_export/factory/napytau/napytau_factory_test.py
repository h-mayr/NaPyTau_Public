import unittest
from unittest.mock import MagicMock, patch

from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity
from napytau.util.model.value_error_pair import ValueErrorPair


class NapytauFactoryUnitTest(unittest.TestCase):
    def test_usesTheJsonFormatServiceToValidateTheDataWhenCreatingADataset(self):
        """Uses the JSON format service to validate the data when creating a dataset"""
        napytau_format_json_service_module_mock = MagicMock()
        napytau_format_json_service_mock = MagicMock()
        napytau_format_json_service_module_mock.NapytauFormatJsonService = (
            napytau_format_json_service_mock
        )

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": napytau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.factory.napytau.napytau_factory import (
                NapyTauFactory,
            )

            raw_json_data = {
                "relativeVelocity": 1,
                "relativeVelocityError": 0.1,
                "datapoints": [],
            }

            NapyTauFactory.create_dataset(raw_json_data)

            self.assertEqual(
                napytau_format_json_service_mock.mock_calls[0].args[0],
                {
                    "relativeVelocity": 1,
                    "relativeVelocityError": 0.1,
                    "datapoints": [],
                },
            )

    def test_canCreateADatasetFromDataWithoutFeedingIntensities(self):
        """Can create a dataset from data without feeding intensities"""

        napytau_format_json_service_module_mock = MagicMock()
        napytau_format_json_service_mock = MagicMock()
        napytau_format_json_service_module_mock.NapytauFormatJsonService = (
            napytau_format_json_service_mock
        )

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": napytau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.factory.napytau.napytau_factory import (
                NapyTauFactory,
            )

            raw_json_data = {
                "relativeVelocity": 1,
                "relativeVelocityError": 0.1,
                "datapoints": [
                    {
                        "distance": 1,
                        "distanceError": 0.1,
                        "normalisation": 1,
                        "normalisationError": 0.1,
                        "shiftedIntensity": 1,
                        "shiftedIntensityError": 0.1,
                        "unshiftedIntensity": 1,
                        "unshiftedIntensityError": 0.1,
                    },
                ],
            }

            dataset = NapyTauFactory.create_dataset(raw_json_data)

            self.assertEqual(dataset.relative_velocity.value.velocity, 1)
            self.assertEqual(dataset.relative_velocity.error.velocity, 0.1)
            self.assertEqual(len(dataset.datapoints), 1)
            self.assertEqual(dataset.datapoints[0].distance.value, 1)
            self.assertEqual(dataset.datapoints[0].distance.error, 0.1)
            self.assertEqual(dataset.datapoints[0].normalisation.value, 1)
            self.assertEqual(dataset.datapoints[0].normalisation.error, 0.1)
            self.assertEqual(dataset.datapoints[0].shifted_intensity.value, 1)
            self.assertEqual(dataset.datapoints[0].shifted_intensity.error, 0.1)
            self.assertEqual(dataset.datapoints[0].unshifted_intensity.value, 1)
            self.assertEqual(dataset.datapoints[0].unshifted_intensity.error, 0.1)
            self.assertEqual(dataset.datapoints[0].feeding_shifted_intensity, None)
            self.assertEqual(dataset.datapoints[0].feeding_unshifted_intensity, None)

    def test_canCreateADatasetFromDataWithFeedingIntensities(self):
        """Can create a dataset from data with feeding intensities"""

        napytau_format_json_service_module_mock = MagicMock()
        napytau_format_json_service_mock = MagicMock()
        napytau_format_json_service_module_mock.NapytauFormatJsonService = (
            napytau_format_json_service_mock
        )

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": napytau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.factory.napytau.napytau_factory import (
                NapyTauFactory,
            )

            raw_json_data = {
                "relativeVelocity": 1,
                "relativeVelocityError": 0.1,
                "datapoints": [
                    {
                        "distance": 1,
                        "distanceError": 0.1,
                        "normalisation": 1,
                        "normalisationError": 0.1,
                        "shiftedIntensity": 1,
                        "shiftedIntensityError": 0.1,
                        "unshiftedIntensity": 1,
                        "unshiftedIntensityError": 0.1,
                        "feedingShiftedIntensity": 1,
                        "feedingShiftedIntensityError": 0.1,
                        "feedingUnshiftedIntensity": 1,
                        "feedingUnshiftedIntensityError": 0.1,
                    },
                ],
            }

            dataset = NapyTauFactory.create_dataset(raw_json_data)

            self.assertEqual(dataset.relative_velocity.value.velocity, 1)
            self.assertEqual(dataset.relative_velocity.error.velocity, 0.1)
            self.assertEqual(len(dataset.datapoints), 1)
            self.assertEqual(dataset.datapoints[0].distance.value, 1)
            self.assertEqual(dataset.datapoints[0].distance.error, 0.1)
            self.assertEqual(dataset.datapoints[0].normalisation.value, 1)
            self.assertEqual(dataset.datapoints[0].normalisation.error, 0.1)
            self.assertEqual(dataset.datapoints[0].shifted_intensity.value, 1)
            self.assertEqual(dataset.datapoints[0].shifted_intensity.error, 0.1)
            self.assertEqual(dataset.datapoints[0].unshifted_intensity.value, 1)
            self.assertEqual(dataset.datapoints[0].unshifted_intensity.error, 0.1)
            self.assertEqual(dataset.datapoints[0].feeding_shifted_intensity.value, 1)
            self.assertEqual(dataset.datapoints[0].feeding_shifted_intensity.error, 0.1)
            self.assertEqual(dataset.datapoints[0].feeding_unshifted_intensity.value, 1)
            self.assertEqual(
                dataset.datapoints[0].feeding_unshifted_intensity.error, 0.1
            )

    def test_canEnrichADatasetWithSetupData(self):
        """Can enrich a dataset with setup data"""
        napytau_format_json_service_module_mock = MagicMock()
        napytau_format_json_service_mock = MagicMock()
        napytau_format_json_service_module_mock.NapytauFormatJsonService = (
            napytau_format_json_service_mock
        )

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": napytau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.factory.napytau.napytau_factory import (
                NapyTauFactory,
            )

            dataset = DataSet(
                ValueErrorPair(
                    RelativeVelocity(1),
                    RelativeVelocity(0.1),
                ),
                DatapointCollection(
                    [
                        Datapoint(
                            distance=ValueErrorPair(1.0, 0.1),
                        )
                    ]
                ),
            )

            raw_setup_data = {
                "tauFactor": 1.0,
                "polynomialCount": 2,
                "datapointSetups": [
                    {
                        "distance": 1.0,
                        "active": True,
                    },
                ],
            }

            dataset = NapyTauFactory.enrich_dataset(dataset, raw_setup_data)

            self.assertEqual(dataset.tau_factor, 1.0)
            self.assertEqual(dataset.polynomial_count, 2)

            datapoint = dataset.datapoints[0]

            self.assertTrue(datapoint.active)


if __name__ == "__main__":
    unittest.main()

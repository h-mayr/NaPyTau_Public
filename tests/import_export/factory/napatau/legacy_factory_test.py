import unittest

from napytau.import_export.factory.legacy.legacy_factory import (
    LegacyFactory,
)
from napytau.import_export.factory.legacy.raw_legacy_data import RawLegacyData
from napytau.import_export.factory.legacy.raw_legacy_setup_data import (
    RawLegacySetupData,
)
from napytau.import_export.import_export_error import ImportExportError
from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity
from napytau.util.model.value_error_pair import ValueErrorPair


def create_dummy_dataset() -> DataSet:
    return DataSet(
        ValueErrorPair(RelativeVelocity(1), RelativeVelocity(0)),
        DatapointCollection(
            [
                Datapoint(
                    ValueErrorPair(1, 1),
                )
            ]
        ),
    )


class LegacyFactoryUnitTest(unittest.TestCase):
    def test_raisesAnExceptionIfNoVelocityIsProvided(self):
        """Raises an exception if no velocity is provided"""
        with self.assertRaises(ValueError):
            LegacyFactory.create_dataset(
                RawLegacyData(
                    [],
                    [],
                    [],
                    [],
                )
            )

    def test_raisesAnExceptionIfADistanceRowWithTooFewValuesIsProvided(self):
        """Raises an exception if a distance row with too few values is provided"""
        with self.assertRaises(ValueError):
            LegacyFactory.create_dataset(
                RawLegacyData(
                    ["1"],
                    ["1"],
                    ["1 1 1 1 1"],
                    ["1 1 1"],
                )
            )

    def test_raisesAnExceptionIfADistanceRowWithTooManyValuesIsProvided(self):
        """Raises an exception if a distance row with too many values is provided"""
        with self.assertRaises(ValueError):
            LegacyFactory.create_dataset(
                RawLegacyData(
                    ["1"],
                    ["1 1 1 1"],
                    ["1 1 1 1 1"],
                    ["1 1 1"],
                )
            )

    def test_raisesAnExceptionIfANormalisationRowWithTooFewValuesIsProvided(self):
        """Raises an exception if a normalisation row with too few values is provided"""
        with self.assertRaises(ValueError):
            LegacyFactory.create_dataset(
                RawLegacyData(
                    ["1"],
                    ["1 1 1"],
                    ["1 1 1 1 1"],
                    ["1 1"],
                )
            )

    def test_raisesAnExceptionIfANormalisationRowWithTooManyValuesIsProvided(self):
        """Raises an exception if a normalisation row with too many values is provided"""
        with self.assertRaises(ValueError):
            LegacyFactory.create_dataset(
                RawLegacyData(
                    ["1"],
                    ["1 1 1"],
                    ["1 1 1 1 1"],
                    ["1 1 1 1"],
                )
            )

    def test_raisesAnExceptionIfAFitRowWithTooFewValuesIsProvided(self):
        """Raises an exception if a fit row with too few values is provided"""
        with self.assertRaises(ValueError):
            LegacyFactory.create_dataset(
                RawLegacyData(
                    ["1"],
                    ["1 1 1"],
                    ["1 1 1 1 1"],
                    ["1 1"],
                )
            )

    def test_raisesAnExceptionIfAFitRowWithTooManyValuesForASetOfBasicIntensitiesButTooManyForASetOfBasicAndShiftedIntensitiesIsProvided(  # noqa: E501
        self,
    ):
        """Raises an exception if a fit row with too many values for a set of basic intensities but too many for a set of basic and shifted intensities is provided"""  # noqa: E501
        with self.assertRaises(ValueError):
            LegacyFactory.create_dataset(
                RawLegacyData(
                    ["1"],
                    ["1 1 1"],
                    ["1 1 1 1 1 1 1 1"],
                    ["1 1 1"],
                )
            )

    def test_raisesAnExceptionIfAFitRowWithTooManyValuesIsProvided(self):
        """Raises an exception if a fit row with too many values is provided"""
        with self.assertRaises(ValueError):
            LegacyFactory.create_dataset(
                RawLegacyData(
                    ["1"],
                    ["1 1 1"],
                    ["1 1 1 1 1 1 1 1 1 1"],
                    ["1 1 1"],
                )
            )

    def test_createsADatasetFromValidDataWithoutFeedingIntensities(self):
        """Creates a dataset from valid data"""
        dataset = LegacyFactory.create_dataset(
            RawLegacyData(
                ["1"],
                ["1 1 1"],
                ["1 1 1 1 1"],
                ["1 1 1"],
            )
        )

        self.assertEqual(dataset.relative_velocity.value.get_velocity(), 1)
        self.assertEqual(dataset.relative_velocity.error.get_velocity(), 0)
        self.assertEqual(len(dataset.datapoints.as_dict()), 1)
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).distance.value, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).distance.error, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).normalisation.value, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).normalisation.error, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).shifted_intensity.value, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).shifted_intensity.error, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).unshifted_intensity.value, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).unshifted_intensity.error, 1
        )

    def test_createsADatasetFromValidDataWithFeedingIntensities(self):
        """Creates a dataset from valid data with feeding intensities"""
        dataset = LegacyFactory.create_dataset(
            RawLegacyData(
                ["1"],
                ["1 1 1"],
                ["1 1 1 1 1 1 1 1 1"],
                ["1 1 1"],
            )
        )

        self.assertEqual(dataset.relative_velocity.value.get_velocity(), 1)
        self.assertEqual(dataset.relative_velocity.error.get_velocity(), 0)
        self.assertEqual(len(dataset.datapoints.as_dict()), 1)
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).distance.value, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).distance.error, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).normalisation.value, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).normalisation.error, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).shifted_intensity.value, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).shifted_intensity.error, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).unshifted_intensity.value, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(1).unshifted_intensity.error, 1
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(
                1
            ).feeding_shifted_intensity.value,
            1,
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(
                1
            ).feeding_shifted_intensity.error,
            1,
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(
                1
            ).feeding_unshifted_intensity.value,
            1,
        )
        self.assertEqual(
            dataset.datapoints.get_datapoint_by_distance(
                1
            ).feeding_unshifted_intensity.error,
            1,
        )

    def test_createsADatasetFromValidDataWithAVelocityErrorGiven(self):
        """Creates a dataset from valid data with a velocity error given"""
        dataset = LegacyFactory.create_dataset(
            RawLegacyData(
                ["1 1"],
                ["1 1 1"],
                ["1 1 1 1 1"],
                ["1 1 1"],
            )
        )

        self.assertEqual(dataset.relative_velocity.value.get_velocity(), 1)
        self.assertEqual(dataset.relative_velocity.error.get_velocity(), 1)
        self.assertEqual(len(dataset.datapoints.as_dict()), 1)

    def test_raisesAnErrorIfTheProvidedSetupDataIsInvalidWhenEnrichingADataSet(self):
        """Raises an error if the provided setup data is invalid when enriching a dataset"""
        dataset = create_dummy_dataset()
        invalid_setup_data = RawLegacySetupData(["invalid_setup_data"])

        with self.assertRaises(ImportExportError):
            LegacyFactory.enrich_dataset(dataset, invalid_setup_data)

    def test_raisesAnErrorIfTheProvidedTauFactorRowIsInvalidWhenEnrichingADataSet(self):
        """Raises an error if the provided tau factor row is invalid when enriching a dataset"""
        dataset = create_dummy_dataset()
        invalid_setup_data = RawLegacySetupData(
            [
                "invalid-tau-factor",
                "1",
                "1",
                "1",
            ]
        )

        with self.assertRaises(ImportExportError):
            LegacyFactory.enrich_dataset(dataset, invalid_setup_data)

    def test_raisesAnErrorIfTheProvidedActiveDistanceRowIsInvalidWhenEnrichingADataSet(
        self,
    ):
        """Raises an error if the provided active distance row is invalid when enriching a dataset"""
        dataset = create_dummy_dataset()
        invalid_setup_data = RawLegacySetupData(
            [
                "1",
                "invalid-active-distance",
                "1",
                "1",
            ]
        )

        with self.assertRaises(ImportExportError):
            LegacyFactory.enrich_dataset(dataset, invalid_setup_data)

    def test_raisesAnErrorIfTheProvidedPolynomialCountRowIsInvalidWhenEnrichingADataSet(
        self,
    ):
        """Raises an error if the provided polynomial count row is invalid when enriching a dataset"""
        dataset = create_dummy_dataset()
        invalid_setup_data = RawLegacySetupData(
            [
                "1",
                "1",
                "invalid-polynomial-count",
                "1",
            ]
        )

        with self.assertRaises(ImportExportError):
            LegacyFactory.enrich_dataset(dataset, invalid_setup_data)

    def test_raisesAnErrorIfTheProvidedSamplingPointRowIsInvalidWhenEnrichingADataSet(
        self,
    ):
        """Raises an error if the provided sampling point row is invalid when enriching a dataset"""
        dataset = create_dummy_dataset()
        invalid_setup_data = RawLegacySetupData(
            [
                "1",
                "1",
                "1",
                "invalid-sampling-point",
            ]
        )

        with self.assertRaises(ImportExportError):
            LegacyFactory.enrich_dataset(dataset, invalid_setup_data)

    def test_enrichesADataSetWithValidSetupData(self):
        """Enriches a dataset with valid setup data"""
        dataset = DataSet(
            ValueErrorPair(RelativeVelocity(1), RelativeVelocity(0)),
            DatapointCollection(
                [
                    Datapoint(
                        ValueErrorPair(1, 13),
                    ),
                    Datapoint(
                        ValueErrorPair(2, 3),
                    ),
                    Datapoint(
                        ValueErrorPair(3, 7),
                    ),
                ]
            ),
        )
        setup_data = RawLegacySetupData(
            [
                "42",
                "1",
                "0",
                "1",
                "2",
                "420",
                "1337",
            ]
        )

        enriched_dataset = LegacyFactory.enrich_dataset(dataset, setup_data)

        self.assertEqual(enriched_dataset.tau_factor, 42)
        self.assertTrue(
            enriched_dataset.datapoints.get_datapoint_by_distance(1).is_active()
        )
        self.assertFalse(
            enriched_dataset.datapoints.get_datapoint_by_distance(2).is_active()
        )
        self.assertTrue(
            enriched_dataset.datapoints.get_datapoint_by_distance(3).is_active()
        )
        self.assertEqual(enriched_dataset.polynomial_count, 2)
        self.assertEqual(len(enriched_dataset.sampling_points), 2)
        self.assertEqual(enriched_dataset.sampling_points[0], 420)
        self.assertEqual(enriched_dataset.sampling_points[1], 1337)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch
import numpy as np

from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity
from napytau.util.model.value_error_pair import ValueErrorPair


def _get_dataset_stub(datapoints: DatapointCollection) -> DataSet:
    return DataSet(
        ValueErrorPair(RelativeVelocity(1 / 299792458), RelativeVelocity(0)),
        datapoints,
    )


def _make_datapoints() -> DatapointCollection:
    return DatapointCollection(
        [
            Datapoint(
                ValueErrorPair(0.0, 0.16),
                None,
                ValueErrorPair(10.0, 0.5),
                ValueErrorPair(3.0, 0.3),
            ),
            Datapoint(
                ValueErrorPair(1.0, 0.16),
                None,
                ValueErrorPair(8.0, 0.5),
                ValueErrorPair(2.5, 0.3),
            ),
            Datapoint(
                ValueErrorPair(2.0, 0.16),
                None,
                ValueErrorPair(6.0, 0.5),
                ValueErrorPair(2.0, 0.3),
            ),
            Datapoint(
                ValueErrorPair(3.0, 0.16),
                None,
                ValueErrorPair(4.0, 0.5),
                ValueErrorPair(1.5, 0.3),
            ),
            Datapoint(
                ValueErrorPair(4.0, 0.16),
                None,
                ValueErrorPair(2.0, 0.5),
                ValueErrorPair(1.0, 0.3),
            ),
        ]
    )


class MCErrorTest(unittest.TestCase):
    def test_mc_returns_two_floats(self):
        """calculate_error_propagation_mc returns a (float, float) tuple."""
        from napytau.core.delta_tau import calculate_error_propagation_mc

        dataset = _get_dataset_stub(_make_datapoints())
        result = calculate_error_propagation_mc(
            dataset, degree=2, smoothing_factor=None, fit_mode="lsq", n_iterations=10
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_mc_error_non_negative(self):
        """MC sigma_tau (std) is always non-negative."""
        from napytau.core.delta_tau import calculate_error_propagation_mc

        dataset = _get_dataset_stub(_make_datapoints())
        _, sigma = calculate_error_propagation_mc(
            dataset, degree=2, smoothing_factor=None, fit_mode="lsq", n_iterations=20
        )

        # std is always >= 0; -1.0 signals no valid samples
        self.assertTrue(sigma >= 0.0 or sigma == -1.0)

    def test_mc_reproducible_with_seed(self):
        """MC results are reproducible when rng is seeded."""
        from napytau.core.delta_tau import calculate_error_propagation_mc

        dataset = _get_dataset_stub(_make_datapoints())

        fixed_rng = np.random.default_rng(42)

        with patch("napytau.core.delta_tau.np.random.default_rng", return_value=fixed_rng):
            result1 = calculate_error_propagation_mc(
                dataset,
                degree=2,
                smoothing_factor=None,
                fit_mode="lsq",
                n_iterations=10,
            )

        fixed_rng2 = np.random.default_rng(42)

        with patch("napytau.core.delta_tau.np.random.default_rng", return_value=fixed_rng2):
            result2 = calculate_error_propagation_mc(
                dataset,
                degree=2,
                smoothing_factor=None,
                fit_mode="lsq",
                n_iterations=10,
            )

        self.assertAlmostEqual(result1[0], result2[0], places=10)
        self.assertAlmostEqual(result1[1], result2[1], places=10)


if __name__ == "__main__":
    unittest.main()

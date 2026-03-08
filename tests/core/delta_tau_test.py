import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.util.model.value_error_pair import ValueErrorPair
from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity


def set_up_mocks() -> (MagicMock, MagicMock, MagicMock, MagicMock):
    polynomial_module_mock = MagicMock()
    polynomial_module_mock.evaluate_polynomial_at_measuring_distances = MagicMock()
    polynomial_module_mock.evaluate_differentiated_polynomial_at_measuring_distances = (
        MagicMock()
    )

    zeros_mock = MagicMock()
    numpy_module_mock = MagicMock()
    numpy_module_mock.zeros = zeros_mock
    numpy_module_mock.diag = MagicMock()
    numpy_module_mock.linalg.pinv = MagicMock()
    numpy_module_mock.power = MagicMock()

    # used actual implementation as these are either data types or functions used for testing only
    numpy_module_mock.array = np.array
    numpy_module_mock.testing = np.testing
    numpy_module_mock.ndarray = np.ndarray
    return polynomial_module_mock, zeros_mock, numpy_module_mock


def _get_dataset_stub(datapoints: DatapointCollection) -> DataSet:
    return DataSet(
        ValueErrorPair(RelativeVelocity(1 / 299792458), RelativeVelocity(0)),
        datapoints,
    )


# Shared knots/degree for test fixtures
_TEST_KNOTS = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
_TEST_DEGREE = 1


class DeltaTauUnitTests(unittest.TestCase):
    @staticmethod
    def test_canCalculateAJacobianMatrixFromDistancesAndCoefficients():
        """Can calculate a Jacobian matrix from distances and coefficients."""
        polynomial_module_mock, zeros_mock, numpy_module_mock = set_up_mocks()

        zeros_mock.return_value = np.array([[0, 0], [0, 0], [0, 0]])
        polynomial_module_mock.evaluate_polynomial_at_measuring_times.side_effect = [
            6,
            3,
            2,
            1,
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.polynomials": polynomial_module_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.delta_tau import calculate_jacobian_matrix

            coefficients = np.array([5, 4])
            datapoints = DatapointCollection(
                [
                    Datapoint(ValueErrorPair(0, 0.16)),
                    Datapoint(ValueErrorPair(1, 0.16)),
                    Datapoint(ValueErrorPair(2, 0.16)),
                ]
            )

            jacobian_matrix = np.array(
                [[3000000, 1000000], [3000000, 1000000], [3000000, 1000000]]
            )

            np.testing.assert_array_equal(
                calculate_jacobian_matrix(
                    _get_dataset_stub(datapoints),
                    coefficients,
                    _TEST_KNOTS,
                    _TEST_DEGREE,
                ),
                jacobian_matrix,
            )

            # Verify knots and degree were forwarded to the polynomial evaluation
            for call in polynomial_module_mock.evaluate_polynomial_at_measuring_times.mock_calls:
                np.testing.assert_array_equal(call.args[2], _TEST_KNOTS)
                assert call.args[3] == _TEST_DEGREE

    def test_canCalculateACovarianceMatrixFromTimesAndCoefficients(self):
        """Can calculate a Covariance matrix from times and coefficients."""
        polynomial_module_mock, zeros_mock, numpy_module_mock = set_up_mocks()

        zeros_mock.return_value = np.array([[0, 0], [0, 0], [0, 0]])
        polynomial_module_mock.evaluate_polynomial_at_measuring_times.side_effect = (
            lambda dataset, coefficients, knots, degree: (np.array([6, 3, 2]))
        )
        numpy_module_mock.power.return_value = np.array([4, 9, 16])
        numpy_module_mock.diag.return_value = np.array(
            [[1 / 4, 0, 0], [0, 1 / 9, 0], [0, 0, 1 / 16]]
        )
        numpy_module_mock.linalg.pinv.return_value = np.array(
            [[-0.13826047, 0.41478141], [0.41478141, -1.24434423]]
        )

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.polynomials": polynomial_module_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.delta_tau import calculate_covariance_matrix

            datapoints = DatapointCollection(
                [
                    Datapoint(ValueErrorPair(0, 0.16), None, ValueErrorPair(0, 2)),
                    Datapoint(ValueErrorPair(1, 0.16), None, ValueErrorPair(0, 3)),
                    Datapoint(ValueErrorPair(2, 0.16), None, ValueErrorPair(0, 4)),
                ]
            )
            coefficients = np.array([5, 4])

            np.testing.assert_array_equal(
                calculate_covariance_matrix(
                    _get_dataset_stub(datapoints),
                    coefficients,
                    _TEST_KNOTS,
                    _TEST_DEGREE,
                ),
                np.array([[-0.13826047, 0.41478141], [0.41478141, -1.24434423]]),
            )

            self.assertEqual(zeros_mock.mock_calls[0].args[0], (3, 2))

            self.assertIsInstance(
                polynomial_module_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                DataSet,
            )
            np.testing.assert_array_equal(
                polynomial_module_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ]
                .args[0]
                .get_datapoints()
                .get_distances()
                .get_values(),
                np.array([0, 1, 2]),
            )

    def test_CanCalculateTheErrorPropagation(self):
        """Can calculate the error propagation using einsum-based delta_p calculation."""
        polynomial_module_mock, zeros_mock, numpy_module_mock = set_up_mocks()

        # evaluate_polynomial_at_measuring_times side_effect:
        # Called twice per coefficient (perturbed, original) × 2 coefficients × 2 jacobian calls
        # = 8 total calls. All return scalars that produce a uniform jacobian.
        polynomial_module_mock.evaluate_polynomial_at_measuring_times.side_effect = [
            6, 3, 2, 1,  # first jacobian call (from calculate_jacobian_matrix explicit)
            6, 3, 2, 1,  # second jacobian call (inside calculate_covariance_matrix)
        ]
        polynomial_module_mock.evaluate_differentiated_polynomial_at_measuring_times.return_value = np.array(
            [4, 4, 4]
        )

        zeros_mock.side_effect = [
            np.array([[0, 0], [0, 0], [0, 0]]),  # first jacobian zeros
            np.array([[0, 0], [0, 0], [0, 0]]),  # second jacobian zeros
        ]
        numpy_module_mock.power.side_effect = [
            np.array([25, 36, 49]),   # unshifted errors^2
            np.array([16, 16, 16]),   # diff_poly^2
            np.array([4, 9, 16]),     # shifted errors^2 (for weight matrix in covariance)
            np.array([4, 5, 6]),      # unshifted values^2 (numerator of poly uncertainty)
            np.array([256, 256, 256]),  # diff_poly^4
            np.array([64, 64, 64]),   # delta_p^2
            np.array([64, 64, 64]),   # diff_poly^3
        ]
        numpy_module_mock.diag.return_value = np.array(
            [[1 / 4, 0, 0], [0, 1 / 9, 0], [0, 0, 1 / 16]]
        )
        numpy_module_mock.linalg.pinv.return_value = np.array(
            [[-0.13826047, 0.41478141], [0.41478141, -1.24434423]]
        )
        numpy_module_mock.einsum = np.einsum
        numpy_module_mock.zeros = zeros_mock

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.polynomials": polynomial_module_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.delta_tau import (
                calculate_error_propagation_terms,
            )

            coefficients: np.array = np.array([5, 4])
            taufactor = 0.4
            datapoints = DatapointCollection(
                [
                    Datapoint(
                        ValueErrorPair(0.0, 0.16),
                        None,
                        ValueErrorPair(0, 2),
                        ValueErrorPair(4, 5),
                    ),
                    Datapoint(
                        ValueErrorPair(1.0, 0.16),
                        None,
                        ValueErrorPair(0, 3),
                        ValueErrorPair(5, 6),
                    ),
                    Datapoint(
                        ValueErrorPair(2.0, 0.16),
                        None,
                        ValueErrorPair(0, 4),
                        ValueErrorPair(6, 7),
                    ),
                ]
            )

            calculate_error_propagation_terms(
                _get_dataset_stub(datapoints),
                coefficients,
                taufactor,
                _TEST_KNOTS,
                _TEST_DEGREE,
            )

            # Verify evaluate_differentiated_polynomial was called with correct args
            self.assertEqual(
                polynomial_module_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                _get_dataset_stub(datapoints),
            )
            np.testing.assert_array_equal(
                polynomial_module_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[1],
                np.array([5, 4]),
            )
            np.testing.assert_array_equal(
                polynomial_module_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[2],
                _TEST_KNOTS,
            )
            self.assertEqual(
                polynomial_module_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[3],
                _TEST_DEGREE,
            )


if __name__ == "__main__":
    unittest.main()

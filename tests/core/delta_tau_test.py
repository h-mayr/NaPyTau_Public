import sys
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
            sys.modules.pop("napytau.core.delta_tau", None)
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
            sys.modules.pop("napytau.core.delta_tau", None)
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
        """
        Verifies the corrected Gaussian error propagation for τᵢ = Iu / P'(t):

            σ(τᵢ)² = σ(Iu_i)²/P'² + Iu_i²·Var[P'(tᵢ)]/P'⁴

        Test setup (degree-1, knots=[0,0,1,1], two datapoints at t=0 and t=1):
          - coefficients = [2, 4]  →  P(t) = 2+2t,  P'(t) = 2 everywhere
          - shifted intensities: [2, 4] ± [0.3, 0.4]
          - unshifted intensities: [3, 5] ± [0.5, 0.7]

        Covariance (diagonal since J=I at endpoints):  diag([0.09, 0.16])
        Var[P'(tᵢ)] = J'@Cov@J'ᵀ = 0.09+0.16 = 0.25  (same at both points)
        term1 = [0.0625, 0.1225]
        term2 = [9·0.25/16, 25·0.25/16] = [0.140625, 0.390625]
        σ(τ) = [√0.203125, √0.513125]
        """
        from napytau.core.delta_tau import calculate_error_propagation_terms

        knots = np.array([0.0, 0.0, 1.0, 1.0])
        degree = 1
        coefficients = np.array([2.0, 4.0])

        datapoints = DatapointCollection(
            [
                Datapoint(
                    ValueErrorPair(0.0, 0.0),
                    None,
                    ValueErrorPair(2.0, 0.3),  # shifted Is=2, σ=0.3
                    ValueErrorPair(3.0, 0.5),  # unshifted Iu=3, σ=0.5
                ),
                Datapoint(
                    ValueErrorPair(1.0, 0.0),
                    None,
                    ValueErrorPair(4.0, 0.4),  # shifted Is=4, σ=0.4
                    ValueErrorPair(5.0, 0.7),  # unshifted Iu=5, σ=0.7
                ),
            ]
        )

        result = calculate_error_propagation_terms(
            _get_dataset_stub(datapoints),
            coefficients,
            knots,
            degree,
        )

        expected = np.sqrt(np.array([0.203125, 0.513125]))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_CanCalculateDerivativeJacobianMatrix(self):
        """
        Verifies calculate_derivative_jacobian_matrix for a degree-1 spline.

        For knots=[0,0,1,1] and degree=1 the two basis functions are:
          B0(t) = 1-t  →  B'0(t) = -1
          B1(t) = t    →  B'1(t) =  1
        so J'[i,:] = [-1, 1] for every evaluation point.
        """
        from napytau.core.delta_tau import calculate_derivative_jacobian_matrix

        knots = np.array([0.0, 0.0, 1.0, 1.0])
        degree = 1
        coefficients = np.array([2.0, 4.0])

        datapoints = DatapointCollection(
            [
                Datapoint(ValueErrorPair(0.0, 0.0)),
                Datapoint(ValueErrorPair(1.0, 0.0)),
            ]
        )

        result = calculate_derivative_jacobian_matrix(
            _get_dataset_stub(datapoints),
            coefficients,
            knots,
            degree,
        )

        expected = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        np.testing.assert_allclose(result, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

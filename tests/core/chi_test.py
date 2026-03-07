import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import numpy.testing
import scipy as sp
from typing import Tuple
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.util.model.value_error_pair import ValueErrorPair
from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity


def set_up_mocks() -> (MagicMock, MagicMock, MagicMock):
    polynomials_mock = MagicMock()
    polynomials_mock.polynomial_sum_at_measuring_times = MagicMock()
    polynomials_mock.differentiated_polynomial_sum_at_measuring_times = MagicMock()

    numpy_module_mock = MagicMock()
    numpy_module_mock.sum = MagicMock()
    numpy_module_mock.power = MagicMock()
    numpy_module_mock.mean = MagicMock()

    scipy_optimize_module_mock = MagicMock()
    scipy_optimize_module_mock.minimize = MagicMock()

    return polynomials_mock, numpy_module_mock, scipy_optimize_module_mock


def _get_dataset_stub(datapoints: DatapointCollection) -> DataSet:
    return DataSet(
        ValueErrorPair(RelativeVelocity(1 / 299792458), RelativeVelocity(0)),
        datapoints,
    )


# Shared test knots/degree
_TEST_KNOTS = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
_TEST_DEGREE = 1


class ChiUnitTest(unittest.TestCase):
    def test_CanCalculateChiForValidData(self):
        """Can calculate chi for valid data"""
        polynomials_mock, numpy_module_mock, scipy_optimize_module_mock = set_up_mocks()

        polynomials_mock.evaluate_polynomial_at_measuring_times.return_value = np.array(
            [5, 15, 57]
        )
        polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.return_value = np.array(
            [4, 20, 72]
        )

        numpy_module_mock.sum.return_value = 628.3486168
        numpy_module_mock.power.side_effect = [
            np.array([4, 18.77777778, 182.25]),
            np.array([0.64, 34.02777778, 388.65306122]),
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.polynomials": polynomials_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.chi import calculate_chi_squared

            coefficients: np.ndarray = np.array([5, 4, 3, 2, 1])
            datapoints = DatapointCollection(
                [
                    Datapoint(
                        ValueErrorPair(0.0, 0.16),
                        None,
                        ValueErrorPair(1, 2),
                        ValueErrorPair(4, 5),
                    ),
                    Datapoint(
                        ValueErrorPair(1.0, 0.16),
                        None,
                        ValueErrorPair(2, 3),
                        ValueErrorPair(5, 6),
                    ),
                    Datapoint(
                        ValueErrorPair(2.0, 0.16),
                        None,
                        ValueErrorPair(3, 4),
                        ValueErrorPair(6, 7),
                    ),
                ]
            )
            t_hyp: float = 2.0
            weight_factor: float = 1.0

            expected_result: float = 628.3486168

            self.assertAlmostEqual(
                calculate_chi_squared(
                    _get_dataset_stub(datapoints),
                    coefficients,
                    t_hyp,
                    weight_factor,
                    _TEST_KNOTS,
                    _TEST_DEGREE,
                ),
                expected_result,
            )

            self.assertEqual(
                len(polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls),
                1,
            )

            self.assertEqual(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                _get_dataset_stub(datapoints),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[1],
                (np.array([5, 4, 3, 2, 1])),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[2],
                _TEST_KNOTS,
            )

            self.assertEqual(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[3],
                _TEST_DEGREE,
            )

            self.assertEqual(
                len(
                    polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls
                ),
                1,
            )

            self.assertEqual(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                _get_dataset_stub(datapoints),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[1],
                (np.array([5, 4, 3, 2, 1])),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[2],
                _TEST_KNOTS,
            )

            self.assertEqual(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[3],
                _TEST_DEGREE,
            )

            self.assertEqual(
                len(numpy_module_mock.sum.mock_calls),
                1,
            )

            np.testing.assert_array_almost_equal(
                numpy_module_mock.sum.mock_calls[0].args[0],
                (np.array([4.64, 52.80555556, 570.90306122])),
            )

            self.assertEqual(
                len(numpy_module_mock.power.mock_calls),
                2,
            )

            np.testing.assert_array_almost_equal(
                numpy_module_mock.power.mock_calls[0].args[0],
                (np.array([-2, -4.33333333, -13.5])),
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[0].args[1],
                2,
            )

            np.testing.assert_array_almost_equal(
                numpy_module_mock.power.mock_calls[1].args[0],
                (np.array([-0.8, -5.83333333, -19.71428571])),
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[1].args[1],
                2,
            )

    def test_CanHandleEmptyDataArraysForChiCalculation(self):
        """Can handle empty data arrays for chi calculation"""
        polynomials_mock, numpy_module_mock, scipy_optimize_module_mock = set_up_mocks()

        # Mocked return values of called functions
        polynomials_mock.evaluate_polynomial_at_measuring_times.return_value = np.array(
            []
        )
        polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.return_value = np.array(
            []
        )

        numpy_module_mock.sum.return_value = 0
        numpy_module_mock.power.side_effect = [np.array([]), np.array([])]

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.polynomials": polynomials_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.chi import calculate_chi_squared

            coefficients: np.ndarray = np.array([])
            datapoints = DatapointCollection([])
            t_hyp: float = 2.0
            weight_factor: float = 1.0

            expected_result: float = 0

            self.assertEqual(
                calculate_chi_squared(
                    _get_dataset_stub(datapoints),
                    coefficients,
                    t_hyp,
                    weight_factor,
                    _TEST_KNOTS,
                    _TEST_DEGREE,
                ),
                expected_result,
            )

            self.assertEqual(
                len(polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls),
                1,
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                (_get_dataset_stub(datapoints)),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[1],
                (np.array([])),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[2],
                _TEST_KNOTS,
            )

            self.assertEqual(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[3],
                _TEST_DEGREE,
            )

            self.assertEqual(
                len(
                    polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls
                ),
                1,
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                (_get_dataset_stub(datapoints)),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[1],
                (np.array([])),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[2],
                _TEST_KNOTS,
            )

            self.assertEqual(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[3],
                _TEST_DEGREE,
            )

            self.assertEqual(
                len(numpy_module_mock.sum.mock_calls),
                1,
            )

            np.testing.assert_array_equal(
                numpy_module_mock.sum.mock_calls[0].args[0],
                np.array([]),
            )

            self.assertEqual(
                len(numpy_module_mock.power.mock_calls),
                2,
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[0].args[0],
                (np.array([])),
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[0].args[1],
                2,
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[1].args[0],
                (np.array([])),
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[1].args[1],
                2,
            )

    def test_CanCalculateChiForASingleDatapoint(self):
        """Can calculate chi for a single datapoint"""
        polynomials_mock, numpy_module_mock, scipy_optimize_module_mock = set_up_mocks()

        # Mocked return values of called functions
        polynomials_mock.evaluate_polynomial_at_measuring_times.return_value = np.array(
            [57]
        )
        polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.return_value = np.array(
            [72]
        )

        numpy_module_mock.sum.return_value = 1608.69444444
        numpy_module_mock.power.side_effect = [
            np.array([348.44444444]),
            np.array([1260.25]),
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.polynomials": polynomials_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.chi import calculate_chi_squared

            coefficients: np.ndarray = np.array([5, 4, 3, 2, 1])
            datapoints = DatapointCollection(
                [
                    Datapoint(
                        ValueErrorPair(2.0, 0.16),
                        None,
                        ValueErrorPair(1, 3),
                        ValueErrorPair(2, 4),
                    ),
                ]
            )
            t_hyp: float = 2.0
            weight_factor: float = 1.0

            expected_result: float = 1608.69444444

            self.assertAlmostEqual(
                calculate_chi_squared(
                    _get_dataset_stub(datapoints),
                    coefficients,
                    t_hyp,
                    weight_factor,
                    _TEST_KNOTS,
                    _TEST_DEGREE,
                ),
                expected_result,
            )

            self.assertEqual(
                len(polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls),
                1,
            )

            self.assertEqual(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                _get_dataset_stub(datapoints),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[1],
                (np.array([5, 4, 3, 2, 1])),
            )

            self.assertEqual(
                len(
                    polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls
                ),
                1,
            )

            self.assertEqual(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                _get_dataset_stub(datapoints),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[1],
                (np.array([5, 4, 3, 2, 1])),
            )

            self.assertEqual(
                len(numpy_module_mock.sum.mock_calls),
                1,
            )

            np.testing.assert_array_almost_equal(
                numpy_module_mock.sum.mock_calls[0].args[0],
                (np.array([1608.69444444])),
            )

            self.assertEqual(
                len(numpy_module_mock.power.mock_calls),
                2,
            )

            np.testing.assert_array_almost_equal(
                numpy_module_mock.power.mock_calls[0].args[0],
                (np.array([-18.66666667])),
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[0].args[1],
                2,
            )

            np.testing.assert_array_almost_equal(
                numpy_module_mock.power.mock_calls[1].args[0],
                (np.array([-35.5])),
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[1].args[1],
                2,
            )

    def test_CanCalculateChiIfTheDenominatorIsZero(self):
        """Can calculate chi if the denominator is zero."""
        polynomials_mock, numpy_module_mock, scipy_optimize_module_mock = set_up_mocks()

        # Mocked return values of called functions
        polynomials_mock.evaluate_polynomial_at_measuring_times.return_value = np.array(
            [5, 15]
        )
        polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.return_value = np.array(
            [4, 20]
        )

        numpy_module_mock.sum.return_value = float("inf")
        numpy_module_mock.power.side_effect = [
            np.array([float("inf"), 169]),
            np.array([float("inf"), 1296]),
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.polynomials": polynomials_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.chi import calculate_chi_squared

            coefficients: np.ndarray = np.array([5, 4, 3, 2, 1])
            datapoints = DatapointCollection(
                [
                    Datapoint(
                        ValueErrorPair(0.0, 0.16),
                        None,
                        ValueErrorPair(1, 0),
                        ValueErrorPair(3, 0),
                    ),
                    Datapoint(
                        ValueErrorPair(1.0, 0.16),
                        None,
                        ValueErrorPair(2, 1),
                        ValueErrorPair(4, 1),
                    ),
                ]
            )
            t_hyp: float = 2.0
            weight_factor: float = 1.0

            expected_result: float = float("inf")

            self.assertAlmostEqual(
                calculate_chi_squared(
                    _get_dataset_stub(datapoints),
                    coefficients,
                    t_hyp,
                    weight_factor,
                    _TEST_KNOTS,
                    _TEST_DEGREE,
                ),
                expected_result,
            )

            self.assertEqual(
                len(polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls),
                1,
            )

            self.assertEqual(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                _get_dataset_stub(datapoints),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[1],
                (np.array([5, 4, 3, 2, 1])),
            )

            self.assertEqual(
                len(
                    polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls
                ),
                1,
            )

            self.assertEqual(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[0],
                _get_dataset_stub(datapoints),
            )

            np.testing.assert_array_equal(
                polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.mock_calls[
                    0
                ].args[1],
                (np.array([5, 4, 3, 2, 1])),
            )

            self.assertEqual(
                len(numpy_module_mock.sum.mock_calls),
                1,
            )

            np.testing.assert_array_equal(
                numpy_module_mock.sum.mock_calls[0].args[0],
                (np.array([float("inf"), 1465])),
            )

            self.assertEqual(
                len(numpy_module_mock.power.mock_calls),
                2,
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[0].args[0],
                (np.array([-float("inf"), -13])),
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[0].args[1],
                2,
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[1].args[0],
                (np.array([-float("inf"), -36])),
            )

            np.testing.assert_array_equal(
                numpy_module_mock.power.mock_calls[1].args[1],
                2,
            )

    def test_CanCalculateChiForNegativeValues(self):
        """Can calculate chi for negative values"""
        polynomials_mock, numpy_module_mock, scipy_optimize_module_mock = set_up_mocks()

        # Mocked return values of called functions
        polynomials_mock.evaluate_polynomial_at_measuring_times.return_value = np.array(
            [-5, -5]
        )
        polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.return_value = np.array(
            [-4, -4]
        )

        numpy_module_mock.sum.return_value = 22.02777778
        numpy_module_mock.power.side_effect = [
            np.array([16, 2.25]),
            np.array([2.77777778, 1]),
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.polynomials": polynomials_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.chi import calculate_chi_squared

            coefficients: np.ndarray = np.array([-5, -4, 3, 2, -1])
            datapoints = DatapointCollection(
                [
                    Datapoint(
                        ValueErrorPair(0.0, 0.16),
                        None,
                        ValueErrorPair(-1, 1),
                        ValueErrorPair(-3, 3),
                    ),
                    Datapoint(
                        ValueErrorPair(1.0, 0.16),
                        None,
                        ValueErrorPair(-2, 2),
                        ValueErrorPair(-4, 4),
                    ),
                ]
            )
            t_hyp: float = 2.0
            weight_factor: float = 1.0

            expected_result: float = 22.02777778

            self.assertAlmostEqual(
                calculate_chi_squared(
                    _get_dataset_stub(datapoints),
                    coefficients,
                    t_hyp,
                    weight_factor,
                    _TEST_KNOTS,
                    _TEST_DEGREE,
                ),
                expected_result,
            )

    def test_CanCalculateChiForAWeightFactorOfZero(self):
        """Can calculate chi for a weight factor of zero"""
        polynomials_mock, numpy_module_mock, scipy_optimize_module_mock = set_up_mocks()

        # Mocked return values of called functions
        polynomials_mock.evaluate_polynomial_at_measuring_times.return_value = np.array(
            [5, 15, 57]
        )
        polynomials_mock.evaluate_differentiated_polynomial_at_measuring_times.return_value = np.array(
            [4, 20, 72]
        )

        numpy_module_mock.sum.return_value = 205.02777778
        numpy_module_mock.power.side_effect = [
            np.array([4, 18.77777778, 182.25]),
            np.array([0.64, 34.02777778, 388.65306122]),
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.polynomials": polynomials_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.chi import calculate_chi_squared

            coefficients: np.ndarray = np.array([5, 4, 3, 2, 1])
            datapoints = DatapointCollection(
                [
                    Datapoint(
                        ValueErrorPair(0.0, 0.16),
                        None,
                        ValueErrorPair(1, 2),
                        ValueErrorPair(4, 5),
                    ),
                    Datapoint(
                        ValueErrorPair(1.0, 0.16),
                        None,
                        ValueErrorPair(2, 3),
                        ValueErrorPair(5, 6),
                    ),
                    Datapoint(
                        ValueErrorPair(2.0, 0.16),
                        None,
                        ValueErrorPair(3, 4),
                        ValueErrorPair(6, 7),
                    ),
                ]
            )
            t_hyp: float = 2.0
            weight_factor: float = 0.0

            expected_result: float = 205.02777778

            self.assertAlmostEqual(
                calculate_chi_squared(
                    _get_dataset_stub(datapoints),
                    coefficients,
                    t_hyp,
                    weight_factor,
                    _TEST_KNOTS,
                    _TEST_DEGREE,
                ),
                expected_result,
            )

    def test_CanOptimizeTHypValue(self):
        """Can optimize t_hyp value"""
        polynomials_mock, numpy_module_mock, scipy_optimize_module_mock = set_up_mocks()

        # Mocked return values of called functions
        scipy_optimize_module_mock.optimize.minimize.return_value = (
            sp.optimize.OptimizeResult(x=2.0)
        )

        numpy_module_mock.mean.return_value = 0

        with patch.dict(
            "sys.modules",
            {
                "scipy": scipy_optimize_module_mock,
                "numpy": numpy_module_mock,
            },
        ):
            from napytau.core.chi import optimize_tau_factor

            initial_coefficients: np.ndarray = np.array([1, 1, 1])
            datapoints = DatapointCollection(
                [
                    Datapoint(
                        ValueErrorPair(0.0, 0.16),
                        None,
                        ValueErrorPair(2, 1),
                        ValueErrorPair(6, 1),
                    ),
                    Datapoint(
                        ValueErrorPair(1.0, 0.16),
                        None,
                        ValueErrorPair(6, 1),
                        ValueErrorPair(10, 1),
                    ),
                ]
            )
            t_hyp_range: Tuple[float, float] = (-5, 5)
            weight_factor: float = 1.0

            expected_t_hyp: float = 2.0

            actual_t_hyp: float = optimize_tau_factor(
                _get_dataset_stub(datapoints),
                weight_factor,
                initial_coefficients,
                t_hyp_range,
                _TEST_KNOTS,
                _TEST_DEGREE,
            )

            self.assertEqual(actual_t_hyp, expected_t_hyp)

            self.assertEqual(
                len(scipy_optimize_module_mock.optimize.minimize.mock_calls), 1
            )

            self.assertTrue(
                callable(
                    scipy_optimize_module_mock.optimize.minimize.mock_calls[0].args[0]
                ),
                """The first argument to minimize should be a callable function""",
            )

            np.testing.assert_array_equal(
                scipy_optimize_module_mock.optimize.minimize.mock_calls[0].kwargs["x0"],
                np.ndarray([0]),
            )

            self.assertEqual(
                scipy_optimize_module_mock.optimize.minimize.mock_calls[0].kwargs[
                    "bounds"
                ],
                [(t_hyp_range[0], t_hyp_range[1])],
            )

            self.assertEqual(len(numpy_module_mock.mean.mock_calls), 1)

            self.assertEqual(numpy_module_mock.mean.mock_calls[0].args[0], (-5, 5))

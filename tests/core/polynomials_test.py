import unittest
from unittest.mock import MagicMock, patch
from napytau.core.errors.polynomial_coefficient_error import (
    PolynomialCoefficientError,
)

import numpy as np
from scipy.interpolate import BSpline

from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.util.model.value_error_pair import ValueErrorPair
from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity


# Linear B-spline test fixtures
# With velocity = 1/c, times = distances / (velocity * c) = distances
# knots = [0, 0, 1, 2, 2], degree = 1, coefficients = [2, 5, 9]
# BSpline([0,0,1,2,2], [2,5,9], 1) evaluates to:
#   t=0 -> 2,  t=1 -> 5,  t=2 -> 9
# Derivative evaluates to:
#   t=0 -> 3 (slope on [0,1]: (5-2)/(1-0))
#   t=1 -> 4 (slope on [1,2]: (9-5)/(2-1), right-side value at knot)
#   t=2 -> 4
_TEST_KNOTS = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
_TEST_COEFFICIENTS = np.array([2.0, 5.0, 9.0])
_TEST_DEGREE = 1


def _get_dataset_stub(datapoints: DatapointCollection) -> DataSet:
    return DataSet(
        ValueErrorPair(RelativeVelocity(1 / 299792458), RelativeVelocity(0)),
        datapoints,
    )


def _three_point_datapoints() -> DatapointCollection:
    return DatapointCollection(
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


class PolynomialsUnitTest(unittest.TestCase):
    @staticmethod
    def test_CanEvaluateAValidSplineAtMeasuringTimes():
        """Can evaluate a valid B-spline at measuring times."""
        from napytau.core.polynomials import evaluate_polynomial_at_measuring_times

        dataset = _get_dataset_stub(_three_point_datapoints())
        expected = BSpline(_TEST_KNOTS, _TEST_COEFFICIENTS, _TEST_DEGREE)(
            np.array([0.0, 1.0, 2.0])
        )

        np.testing.assert_array_almost_equal(
            evaluate_polynomial_at_measuring_times(
                dataset, _TEST_COEFFICIENTS, _TEST_KNOTS, _TEST_DEGREE
            ),
            expected,
        )

    @staticmethod
    def test_CanEvaluateASplineAtMeasuringTimesForASingleDistance():
        """Can evaluate a B-spline at measuring times for a single distance."""
        from napytau.core.polynomials import evaluate_polynomial_at_measuring_times

        datapoints = DatapointCollection(
            [
                Datapoint(
                    ValueErrorPair(1.0, 0.16),
                    None,
                    ValueErrorPair(0, 3),
                    ValueErrorPair(5, 6),
                ),
            ]
        )
        dataset = _get_dataset_stub(datapoints)
        expected = BSpline(_TEST_KNOTS, _TEST_COEFFICIENTS, _TEST_DEGREE)(
            np.array([1.0])
        )

        np.testing.assert_array_almost_equal(
            evaluate_polynomial_at_measuring_times(
                dataset, _TEST_COEFFICIENTS, _TEST_KNOTS, _TEST_DEGREE
            ),
            expected,
        )

    def test_EvaluateSplineRaisesPolynomialCoefficientErrorForEmptyCoefficientArray(
        self,
    ):
        """Evaluate spline raises a PolynomialCoefficientError for an empty coefficient array."""
        from napytau.core.polynomials import evaluate_polynomial_at_measuring_times

        dataset = _get_dataset_stub(_three_point_datapoints())
        coefficients: np.ndarray = np.array([])

        with self.assertRaises(PolynomialCoefficientError):
            evaluate_polynomial_at_measuring_times(
                dataset,
                coefficients,
                _TEST_KNOTS,
                _TEST_DEGREE,
            )

    @staticmethod
    def test_CanEvaluateAValidDifferentiatedSplineAtMeasuringTimes():
        """Can evaluate a valid differentiated B-spline at measuring times."""
        from napytau.core.polynomials import (
            evaluate_differentiated_polynomial_at_measuring_times,
        )

        dataset = _get_dataset_stub(_three_point_datapoints())
        expected = BSpline(_TEST_KNOTS, _TEST_COEFFICIENTS, _TEST_DEGREE).derivative()(
            np.array([0.0, 1.0, 2.0])
        )

        np.testing.assert_array_almost_equal(
            evaluate_differentiated_polynomial_at_measuring_times(
                dataset, _TEST_COEFFICIENTS, _TEST_KNOTS, _TEST_DEGREE
            ),
            expected,
        )

    @staticmethod
    def test_CanEvaluateADifferentiatedSplineAtMeasuringTimesForASingleDistance():
        """Can evaluate a differentiated B-spline at measuring times for a single distance."""
        from napytau.core.polynomials import (
            evaluate_differentiated_polynomial_at_measuring_times,
        )

        datapoints = DatapointCollection(
            [
                Datapoint(
                    ValueErrorPair(0.5, 0.16),
                    None,
                    ValueErrorPair(0, 3),
                    ValueErrorPair(5, 6),
                ),
            ]
        )
        dataset = _get_dataset_stub(datapoints)
        expected = BSpline(_TEST_KNOTS, _TEST_COEFFICIENTS, _TEST_DEGREE).derivative()(
            np.array([0.5])
        )

        np.testing.assert_array_almost_equal(
            evaluate_differentiated_polynomial_at_measuring_times(
                dataset, _TEST_COEFFICIENTS, _TEST_KNOTS, _TEST_DEGREE
            ),
            expected,
        )

    def test_EvaluateDifferentiatedSplineRaisesPolynomialCoefficientErrorForEmptyCoefficientArray(
        self,
    ):
        """Evaluate differentiated spline raises a PolynomialCoefficientError for an empty coefficient array."""
        from napytau.core.polynomials import (
            evaluate_differentiated_polynomial_at_measuring_times,
        )

        dataset = _get_dataset_stub(_three_point_datapoints())
        coefficients: np.ndarray = np.array([])

        with self.assertRaises(PolynomialCoefficientError):
            evaluate_differentiated_polynomial_at_measuring_times(
                dataset,
                coefficients,
                _TEST_KNOTS,
                _TEST_DEGREE,
            )

    def test_CalculateCoefficientForFitFallsBackToUnivariateSplineWhenSamplingPointsAreNone(
        self,
    ):
        """calculate_polynomial_coefficients_for_fit falls back to UnivariateSpline when sampling_points is None."""
        from napytau.core.polynomials import calculate_polynomial_coefficients_for_fit

        dataset = _get_dataset_stub(_three_point_datapoints())
        # dataset.sampling_points is None by default

        coefficients, knots = calculate_polynomial_coefficients_for_fit(
            dataset, degree=2
        )
        self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsInstance(knots, np.ndarray)

    def test_CalculateCoefficientForFitFallsBackToUnivariateSplineWhenSamplingPointsAreEmpty(
        self,
    ):
        """calculate_polynomial_coefficients_for_fit falls back to UnivariateSpline when sampling_points is empty."""
        from napytau.core.polynomials import calculate_polynomial_coefficients_for_fit

        dataset = _get_dataset_stub(_three_point_datapoints())
        dataset.set_sampling_points([])

        coefficients, knots = calculate_polynomial_coefficients_for_fit(
            dataset, degree=2
        )
        self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsInstance(knots, np.ndarray)

    def test_CalculateCoefficientForFitReturnsTuple(self):
        """calculate_polynomial_coefficients_for_fit returns (coefficients, knots) tuple."""
        from napytau.core.polynomials import (
            calculate_polynomial_coefficients_for_fit,
        )

        spline_mock = MagicMock()
        spline_mock._eval_args = (
            np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            2,
        )

        dataset = _get_dataset_stub(_three_point_datapoints())
        dataset.set_sampling_points([0.5])

        with patch("napytau.core.polynomials.LSQUnivariateSpline") as mock_spline_cls:
            mock_spline_cls.return_value = spline_mock
            result = calculate_polynomial_coefficients_for_fit(dataset, degree=2)

        assert isinstance(result, tuple)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(
            result[1], np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        )

    def test_CalculateCoefficientForFitUsesUnivariateSplineWhenSmoothingFactorSet(
        self,
    ):
        """calculate_polynomial_coefficients_for_fit uses UnivariateSpline when smoothing_factor is set."""
        from napytau.core.polynomials import calculate_polynomial_coefficients_for_fit

        spline_mock = MagicMock()
        spline_mock._eval_args = (
            np.array([0.0, 0.0, 1.0, 2.0, 2.0]),
            np.array([2.0, 5.0, 9.0]),
            2,
        )

        dataset = _get_dataset_stub(_three_point_datapoints())

        with patch(
            "napytau.core.polynomials.UnivariateSpline"
        ) as mock_univariate, patch(
            "napytau.core.polynomials.LSQUnivariateSpline"
        ) as mock_lsq:
            mock_univariate.return_value = spline_mock
            calculate_polynomial_coefficients_for_fit(
                dataset, degree=2, smoothing_factor=1.0
            )

        mock_univariate.assert_called_once()
        mock_lsq.assert_not_called()


if __name__ == "__main__":
    unittest.main()

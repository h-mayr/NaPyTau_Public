import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from typing import Tuple, Optional


from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.relative_velocity import RelativeVelocity
from napytau.util.model.value_error_pair import ValueErrorPair
from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.dataset import DataSet


def set_up_mocks() -> (MagicMock, MagicMock, MagicMock, MagicMock):
    chi_mock = MagicMock()
    chi_mock.optimize_t_hyp = MagicMock()
    chi_mock.optimize_coefficients = MagicMock()

    tau_mock = MagicMock()
    tau_mock.calculate_tau_i_values = MagicMock()

    delta_tau_mock = MagicMock()
    delta_tau_mock.calculate_error_propagation_terms = MagicMock()

    tau_final_mock = MagicMock()
    tau_final_mock.calculate_tau_final = MagicMock()

    polynomial_mock = MagicMock()
    polynomial_mock.calculate_polynomial_coefficients_for_fit = MagicMock()
    polynomial_mock.calculate_polynomial_coefficients_for_tau_factor = MagicMock()

    return chi_mock, tau_mock, delta_tau_mock, tau_final_mock, polynomial_mock


def _get_dataset_stub(datapoints: DatapointCollection) -> DataSet:
    return DataSet(
        ValueErrorPair(RelativeVelocity(1 / 299792458), RelativeVelocity(0)),
        datapoints,
    )


# Shared test knots returned by mocked fit functions
_MOCK_COEFFICIENTS = np.array([1, 1, 1])
_MOCK_KNOTS = np.array([0.0, 0.0, 0.5, 1.0, 1.0])


class CoreUnitTest(unittest.TestCase):
    def test_CanCalculateALifetime(self):
        """Can calculate a lifetime"""
        chi_mock, tau_mock, delta_tau_mock, tau_final_mock, polynomial_mock = (
            set_up_mocks()
        )

        # Mocked return values of called functions
        chi_mock.optimize_tau_factor.return_value = 2.0
        chi_mock.optimize_coefficients.return_value = (np.array([2, 3, 1]), 2.0)

        tau_mock.calculate_tau_i_values.return_value = np.array([3, 1.66666667])

        delta_tau_mock.calculate_error_propagation_terms.return_value = np.array(
            [0.6, 0.2]
        )

        # Return a (coefficients, knots) tuple — new API
        polynomial_mock.calculate_polynomial_coefficients_for_tau_factor.return_value = (
            _MOCK_COEFFICIENTS,
            _MOCK_KNOTS,
        )

        tau_final_mock.calculate_tau_final.return_value = (1.8, 0.18973666)

        with patch.dict(
            "sys.modules",
            {
                "napytau.core.chi": chi_mock,
                "napytau.core.tau": tau_mock,
                "napytau.core.delta_tau": delta_tau_mock,
                "napytau.core.tau_final": tau_final_mock,
                "napytau.core.polynomials": polynomial_mock,
            },
        ):
            from napytau.core.core import calculate_lifetime_for_custom_tau_factor

            dataset = DataSet(
                ValueErrorPair(RelativeVelocity(0.4), RelativeVelocity(0.02)),
                DatapointCollection(
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
                ),
            )
            custom_t_hyp_estimate: Optional[float] = 1.0

            actual_result: Tuple[float, float] = (
                calculate_lifetime_for_custom_tau_factor(
                    dataset,
                    custom_t_hyp_estimate,
                    2,
                )
            )

            self.assertAlmostEqual(actual_result[0], 1.8)

            self.assertAlmostEqual(actual_result[1], 0.18973666)

            self.assertEqual(
                len(
                    polynomial_mock.calculate_polynomial_coefficients_for_tau_factor.mock_calls
                ),
                1,
            )
            self.assertEqual(
                polynomial_mock.calculate_polynomial_coefficients_for_tau_factor.mock_calls[
                    0
                ].args[0],
                dataset,
            )
            self.assertEqual(
                polynomial_mock.calculate_polynomial_coefficients_for_tau_factor.mock_calls[
                    0
                ].args[1],
                1.0,
            )
            self.assertEqual(
                polynomial_mock.calculate_polynomial_coefficients_for_tau_factor.mock_calls[
                    0
                ].args[2],
                2,
            )

            self.assertEqual(len(tau_mock.calculate_tau_i_values.mock_calls), 1)

            np.testing.assert_array_equal(
                tau_mock.calculate_tau_i_values.mock_calls[0].args[0],
                dataset,
            )

            np.testing.assert_array_equal(
                tau_mock.calculate_tau_i_values.mock_calls[0].args[1],
                _MOCK_COEFFICIENTS,
            )

            # Verify knots and degree are forwarded to calculate_tau_i_values
            np.testing.assert_array_equal(
                tau_mock.calculate_tau_i_values.mock_calls[0].args[2],
                _MOCK_KNOTS,
            )

            self.assertEqual(
                tau_mock.calculate_tau_i_values.mock_calls[0].args[3],
                2,
            )

            self.assertEqual(
                len(delta_tau_mock.calculate_error_propagation_terms.mock_calls), 1
            )

            np.testing.assert_array_equal(
                delta_tau_mock.calculate_error_propagation_terms.mock_calls[0].args[0],
                dataset,
            )

            np.testing.assert_array_equal(
                delta_tau_mock.calculate_error_propagation_terms.mock_calls[0].args[1],
                _MOCK_COEFFICIENTS,
            )

            self.assertEqual(
                delta_tau_mock.calculate_error_propagation_terms.mock_calls[0].args[2],
                1.0,
            )

            # Verify knots and degree are forwarded to calculate_error_propagation_terms
            np.testing.assert_array_equal(
                delta_tau_mock.calculate_error_propagation_terms.mock_calls[0].args[3],
                _MOCK_KNOTS,
            )

            self.assertEqual(
                delta_tau_mock.calculate_error_propagation_terms.mock_calls[0].args[4],
                2,
            )

            self.assertEqual(len(tau_final_mock.calculate_tau_final.mock_calls), 1)

            np.testing.assert_array_equal(
                tau_final_mock.calculate_tau_final.mock_calls[0].args[0],
                np.array([3, 1.66666667]),
            )

            np.testing.assert_array_equal(
                tau_final_mock.calculate_tau_final.mock_calls[0].args[1],
                np.array([0.6, 0.2]),
            )

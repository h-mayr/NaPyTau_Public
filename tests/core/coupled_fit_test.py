import unittest
import numpy as np
from scipy.interpolate import BSpline

from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity
from napytau.util.model.value_error_pair import ValueErrorPair


def _get_dataset_stub(datapoints: DatapointCollection) -> DataSet:
    # velocity = 1/c so that times = distances
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
                ValueErrorPair(10.0, 1.0),
                ValueErrorPair(3.0, 0.5),
            ),
            Datapoint(
                ValueErrorPair(1.0, 0.16),
                None,
                ValueErrorPair(8.0, 1.0),
                ValueErrorPair(2.5, 0.5),
            ),
            Datapoint(
                ValueErrorPair(2.0, 0.16),
                None,
                ValueErrorPair(6.0, 1.0),
                ValueErrorPair(2.0, 0.5),
            ),
            Datapoint(
                ValueErrorPair(3.0, 0.16),
                None,
                ValueErrorPair(4.0, 1.0),
                ValueErrorPair(1.5, 0.5),
            ),
            Datapoint(
                ValueErrorPair(4.0, 0.16),
                None,
                ValueErrorPair(2.0, 1.0),
                ValueErrorPair(1.0, 0.5),
            ),
        ]
    )


class CoupledFitTest(unittest.TestCase):
    def test_coupled_fit_returns_correct_shape(self):
        """calculate_polynomial_coefficients_for_coupled_fit returns ndarray pair."""
        from napytau.core.polynomials import (
            calculate_polynomial_coefficients_for_coupled_fit,
        )

        dataset = _get_dataset_stub(_make_datapoints())
        coefficients, knots = calculate_polynomial_coefficients_for_coupled_fit(
            dataset, tau_factor=2.0, degree=2
        )

        self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsInstance(knots, np.ndarray)
        self.assertGreater(len(coefficients), 0)
        self.assertGreater(len(knots), 0)
        # n_coeffs = len(knots) - degree - 1
        self.assertEqual(len(coefficients), len(knots) - 2 - 1)

    def test_coupled_fit_normal_equations_satisfied(self):
        """Coupled fit solution satisfies A @ c ≈ Y (normal equations)."""
        from napytau.core.polynomials import (
            calculate_polynomial_coefficients_for_coupled_fit,
        )

        dataset = _get_dataset_stub(_make_datapoints())
        tau_factor = 2.0
        degree = 2
        weight_factor = 1.0

        coefficients, knots = calculate_polynomial_coefficients_for_coupled_fit(
            dataset, tau_factor=tau_factor, degree=degree, weight_factor=weight_factor
        )

        # Reconstruct A and Y to verify A @ c ≈ Y
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        datapoints = dataset.get_datapoints()
        i_sh = np.array(datapoints.get_shifted_intensities().get_values())
        sigma_sh = np.array(datapoints.get_shifted_intensities().get_errors())
        i_us = np.array(datapoints.get_unshifted_intensities().get_values())
        sigma_us = np.array(datapoints.get_unshifted_intensities().get_errors())

        n = len(times)
        n_coeffs = len(knots) - degree - 1
        b_matrix = np.zeros((n, n_coeffs))
        db_matrix = np.zeros((n, n_coeffs))
        for j in range(n_coeffs):
            e_j = np.zeros(n_coeffs)
            e_j[j] = 1.0
            spl = BSpline(knots, e_j, degree)
            b_matrix[:, j] = np.asarray(spl(times))
            db_matrix[:, j] = np.asarray(spl.derivative()(times))

        fac1 = (2.0 - weight_factor) / sigma_sh**2
        fac2 = weight_factor * tau_factor / sigma_us**2
        a_matrix = (
            b_matrix.T @ np.diag(fac1) @ b_matrix
            + db_matrix.T @ np.diag(fac2 * tau_factor) @ db_matrix
        )
        y_vector = b_matrix.T @ (fac1 * i_sh) + db_matrix.T @ (fac2 * i_us)

        np.testing.assert_allclose(a_matrix @ coefficients, y_vector, atol=1e-8)

    def test_coupled_fit_with_zero_weight_uses_shifted_only(self):
        """Coupled fit with weight_factor=0 produces the same knots as standard LSQ."""
        from napytau.core.polynomials import (
            calculate_polynomial_coefficients_for_coupled_fit,
            calculate_polynomial_coefficients_for_fit,
        )

        dataset = _get_dataset_stub(_make_datapoints())
        degree = 2

        _, knots_coupled = calculate_polynomial_coefficients_for_coupled_fit(
            dataset, tau_factor=2.0, degree=degree, weight_factor=0.0
        )
        _, knots_lsq = calculate_polynomial_coefficients_for_fit(dataset, degree=degree)

        np.testing.assert_array_almost_equal(knots_coupled, knots_lsq)


if __name__ == "__main__":
    unittest.main()

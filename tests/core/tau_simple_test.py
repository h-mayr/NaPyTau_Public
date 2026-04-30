import unittest
from unittest.mock import patch
import numpy as np
from napytau.core.tau_simple import calc_tau_simple


class TauSimpleUnitTests(unittest.TestCase):
    def test_calc_tau_simple(self):
        """Test tau_simple with values from 132Te dataset."""
        intensities_unshifted = np.array(
            [
                1177.15468035,
                1094.40000617,
                1310.11130973,
                1384.08781108,
                1270.4635517,
                1138.74900115,
                1127.98060615,
                1197.39865685,
                1056.74890986,
                757.00387421,
                626.36091347,
                306.53948207,
            ]
        )
        intensities_unshifted_unc = np.array(
            [
                80.98796399,
                73.73576285,
                73.72799963,
                77.74673988,
                67.90963276,
                61.94946441,
                70.47992774,
                79.60007913,
                71.82884912,
                60.20154397,
                60.13362304,
                85.50112313,
            ]
        )
        intensities_shifted = np.array(
            [
                269.55163934,
                368.04142396,
                128.33227064,
                133.05031147,
                284.86936634,
                138.26256226,
                329.70887352,
                229.27332962,
                833.34860094,
                580.09553565,
                785.13065263,
                1003.70968423,
            ]
        )
        intensities_shifted_unc = np.array(
            [
                84.39238958,
                77.1037839,
                73.87649544,
                76.90126852,
                69.31586135,
                64.20747153,
                73.97034696,
                82.65020657,
                82.68776247,
                70.84063257,
                72.96506007,
                115.24693167,
            ]
        )
        flight_times = np.array(
            [
                1.82184202e-12,
                3.83668109e-12,
                6.07538868e-12,
                1.05527367e-11,
                1.50300623e-11,
                2.51042129e-11,
                6.54004570e-11,
                8.77873986e-11,
                1.24725592e-10,
                1.80692700e-10,
                2.47853950e-10,
                3.37399276e-10,
            ]
        )

        with patch("napytau.core.tau_simple._prepare_intensity_arrays") as mock_prep:
            mock_prep.return_value = (
                intensities_unshifted,
                intensities_unshifted_unc,
                intensities_shifted,
                intensities_shifted_unc,
                flight_times,
            )

            result = calc_tau_simple(dataset=None)

        self.assertAlmostEqual(result[0], 3.57017089e-10, delta=1e-14)
        self.assertAlmostEqual(result[1], 1.32525943e-10, delta=1e-14)

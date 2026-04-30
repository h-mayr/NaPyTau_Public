import unittest
from napytau.import_export.model.datapoint import Datapoint
from napytau.util.model.value_error_pair import ValueErrorPair


class DatapointUnitTest(unittest.TestCase):
    def test_raisesAnExceptionIfNormalisationIsAccessedBeforeInitialization(self):
        """Raise an exception if normalisation is accessed before initialization."""
        datapoint = Datapoint(ValueErrorPair(1.0, 0.1))
        with self.assertRaises(Exception):
            datapoint.get_normalisation()

    def test_raisesAnExceptionIfIntensityIsAccessedBeforeInitialization(self):
        """Raise an exception if intensity is accessed before initialization."""
        datapoint = Datapoint(ValueErrorPair(1.0, 0.1))
        with self.assertRaises(Exception):
            datapoint.get_intensity()

    def test_raisesAnExceptionIfTauIsAccessedBeforeInitialization(self):
        """Raise an exception if tau is accessed before initialization."""
        datapoint = Datapoint(ValueErrorPair(1.0, 0.1))
        with self.assertRaises(Exception):
            datapoint.get_tau()


if __name__ == "__main__":
    unittest.main()

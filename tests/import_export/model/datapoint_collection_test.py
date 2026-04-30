import unittest

from napytau.import_export.model.datapoint import Datapoint
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.util.model.ValueErrorPairCollection import ValueErrorPairCollection
from napytau.util.model.value_error_pair import ValueErrorPair


class DatapointCollectionUnitTest(unittest.TestCase):
    def test_canDetermineItsLength(self):
        """Can determine its length"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                ),
            ]
        )

        self.assertEqual(len(collection), 2)

    def test_canBeIteratedOver(self):
        """Can be iterated over"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                ),
            ]
        )

        for datapoint in collection:
            self.assertIsInstance(datapoint, Datapoint)

    def test_canBeIndexed(self):
        """Can be indexed"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                ),
            ]
        )

        self.assertIsInstance(collection[0], Datapoint)
        self.assertEqual(collection[0].distance.value, 12.12)

    def test_distinguishesDistancesToAReasonablePrecision(self):
        """Can distinguish two datapoints based on their distance values to a reasonable precision"""
        datapoints = [
            Datapoint(
                distance=ValueErrorPair(17.1381257194, 0.1),
            ),
            Datapoint(
                distance=ValueErrorPair(17.1381257195, 0.1),
            ),
            Datapoint(
                distance=ValueErrorPair(17.1381257195, 0.1),
            ),
        ]

        collection = DatapointCollection(datapoints)

        self.assertEqual(len(collection.elements), 2)

    def test_canBeTurnedIntoADict(self):
        """Can be created from a list of datapoints"""
        datapoints = [
            Datapoint(
                distance=ValueErrorPair(12.12, 0.1),
            ),
        ]

        collection = DatapointCollection(datapoints)

        self.assertEqual(collection.as_dict(), {276701161105641484: datapoints[0]})

    def test_canFilterOutValuesBasedOnTheProvidedFilterFunction(self):
        """Can filter out values based on the provided filter function"""
        datapoints = [
            Datapoint(
                distance=ValueErrorPair(12.12, 0.1),
            ),
            Datapoint(
                distance=ValueErrorPair(12.11, 0.1),
            ),
        ]

        collection = DatapointCollection(datapoints)

        filtered = collection.filter(lambda x: x.distance.value == 12.12)

        self.assertEqual(len(filtered.elements), 1)
        self.assertEqual(filtered.elements[hash(12.12)], datapoints[0])

    def test_canAddADatapoint(self):
        """Can add a datapoint to the collection"""
        collection = DatapointCollection(list())

        datapoint = Datapoint(
            distance=ValueErrorPair(12.12, 0.1),
        )

        collection.add_datapoint(datapoint)

        self.assertEqual(collection.elements[hash(12.12)], datapoint)

    def test_overridesDatapointWithSameDistance(self):
        """Overrides a datapoint with the same distance"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                )
            ]
        )

        new_datapoint = Datapoint(
            distance=ValueErrorPair(12.12, 0.1),
        )

        collection.add_datapoint(new_datapoint)

        self.assertEqual(collection.elements[hash(12.12)], new_datapoint)

    def test_canRetrieveADatapointByDistance(self):
        """Can retrieve a datapoint by its distance"""
        datapoint = Datapoint(
            distance=ValueErrorPair(12.12, 0.1),
        )
        collection = DatapointCollection([datapoint])

        self.assertEqual(
            collection.get_datapoint_by_distance(12.12),
            datapoint,
        )

    def test_raisesErrorWhenDatapointNotFound(self):
        """Raises an error when a datapoint is not found"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                )
            ]
        )

        with self.assertRaises(ValueError):
            collection.get_datapoint_by_distance(12.13)

    def test_canRetrieveDistances(self):
        """Can retrieve distances"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                ),
            ]
        )

        self.assertEqual(
            collection.get_distances(),
            ValueErrorPairCollection(
                [ValueErrorPair(12.12, 0.1), ValueErrorPair(12.13, 0.1)]
            ),
        )

    def test_canRetrieveNormalisations(self):
        """Can retrieve normalisations"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                    normalisation=ValueErrorPair(1.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                    normalisation=ValueErrorPair(2.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.14, 0.1),
                ),
            ]
        )

        self.assertEqual(
            collection.get_normalisations(),
            ValueErrorPairCollection(
                [ValueErrorPair(1.0, 0.1), ValueErrorPair(2.0, 0.1)]
            ),
        )

    def test_canRetrieveShiftedIntensities(self):
        """Can retrieve shifted intensities"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                    shifted_intensity=ValueErrorPair(1.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                    shifted_intensity=ValueErrorPair(2.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.14, 0.1),
                ),
            ]
        )

        self.assertEqual(
            collection.get_shifted_intensities(),
            ValueErrorPairCollection(
                [ValueErrorPair(1.0, 0.1), ValueErrorPair(2.0, 0.1)]
            ),
        )

    def test_canRetrieveUnshiftedIntensities(self):
        """Can retrieve unshifted intensities"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                    unshifted_intensity=ValueErrorPair(1.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                    unshifted_intensity=ValueErrorPair(2.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.14, 0.1),
                ),
            ]
        )

        self.assertEqual(
            collection.get_unshifted_intensities(),
            ValueErrorPairCollection(
                [ValueErrorPair(1.0, 0.1), ValueErrorPair(2.0, 0.1)]
            ),
        )

    def test_canRetrieveFeedingShiftedIntensities(self):
        """Can retrieve feeding shifted intensities"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                    feeding_shifted_intensity=ValueErrorPair(1.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                    feeding_shifted_intensity=ValueErrorPair(2.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.14, 0.1),
                ),
            ]
        )

        self.assertEqual(
            collection.get_feeding_shifted_intensities(),
            ValueErrorPairCollection(
                [ValueErrorPair(1.0, 0.1), ValueErrorPair(2.0, 0.1)]
            ),
        )

    def test_canRetrieveFeedingUnshiftedIntensities(self):
        """Can retrieve feeding unshifted intensities"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                    feeding_unshifted_intensity=ValueErrorPair(1.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                    feeding_unshifted_intensity=ValueErrorPair(2.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.14, 0.1),
                ),
            ]
        )

        self.assertEqual(
            collection.get_feeding_unshifted_intensities(),
            ValueErrorPairCollection(
                [ValueErrorPair(1.0, 0.1), ValueErrorPair(2.0, 0.1)]
            ),
        )

    def test_canRetrieveTaus(self):
        """Can retrieve taus"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                    tau=ValueErrorPair(1.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                    tau=ValueErrorPair(2.0, 0.1),
                ),
                Datapoint(
                    distance=ValueErrorPair(12.14, 0.1),
                ),
            ]
        )

        self.assertEqual(
            collection.get_taus(),
            ValueErrorPairCollection(
                [ValueErrorPair(1.0, 0.1), ValueErrorPair(2.0, 0.1)]
            ),
        )

    def test_canRetrieveActiveDatapoints(self):
        """Can retrieve active datapoints"""
        collection = DatapointCollection(
            [
                Datapoint(
                    distance=ValueErrorPair(12.12, 0.1),
                    active=True,
                ),
                Datapoint(
                    distance=ValueErrorPair(12.13, 0.1),
                    active=False,
                ),
                Datapoint(
                    distance=ValueErrorPair(12.14, 0.1),
                    active=True,
                ),
            ]
        )

        self.assertEqual(
            list(collection.get_active_datapoints().as_dict().values()),
            [collection.elements[hash(12.12)], collection.elements[hash(12.14)]],
        )


if __name__ == "__main__":
    unittest.main()

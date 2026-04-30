import unittest
from pathlib import PurePath

from napytau.import_export.crawler.legacy_setup_files import LegacySetupFiles
from napytau.import_export.import_export_error import ImportExportError


class LegacySetupFilesUnitTest(unittest.TestCase):
    def test_raisesAnErrorIfAnyFileIsNotProvided(self):
        """Raises an error if any file is not provided."""
        with self.assertRaises(ImportExportError):
            LegacySetupFiles.create_from_file_names([])
        with self.assertRaises(ImportExportError):
            LegacySetupFiles.create_from_file_names(
                [
                    PurePath("v_c"),
                    PurePath("test.fit"),
                    PurePath("norm.fac"),
                ]
            )
        with self.assertRaises(ImportExportError):
            LegacySetupFiles.create_from_file_names(
                [
                    PurePath("distances.dat"),
                    PurePath("test.fit"),
                    PurePath("norm.fac"),
                ]
            )
        with self.assertRaises(ImportExportError):
            LegacySetupFiles.create_from_file_names(
                [
                    PurePath("distances.dat"),
                    PurePath("v_c"),
                    PurePath("norm.fac"),
                ]
            )
        with self.assertRaises(ImportExportError):
            LegacySetupFiles.create_from_file_names(
                [
                    PurePath("distances.dat"),
                    PurePath("v_c"),
                    PurePath("test.fit"),
                ]
            )

    def test_canBeCreatedFromAListOfFileNames(self):
        """Can be created from a list of file names."""
        file_names = [
            PurePath("distances.dat"),
            PurePath("v_c"),
            PurePath("test.fit"),
            PurePath("norm.fac"),
        ]
        setup_files = LegacySetupFiles.create_from_file_names(file_names)
        self.assertEqual(setup_files.distances_file, PurePath("distances.dat"))
        self.assertEqual(setup_files.velocity_file, PurePath("v_c"))
        self.assertEqual(setup_files.fit_file, PurePath("test.fit"))
        self.assertEqual(setup_files.normalisation_file, PurePath("norm.fac"))


if __name__ == "__main__":
    unittest.main()

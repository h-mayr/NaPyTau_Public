import unittest
from pathlib import PurePath
from unittest.mock import MagicMock, patch

from napytau.import_export.crawler.legacy_setup_files import LegacySetupFiles
from napytau.import_export.factory.legacy.raw_legacy_setup_data import (
    RawLegacySetupData,
)
from napytau.import_export.model.datapoint_collection import DatapointCollection
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.model.relative_velocity import RelativeVelocity
from napytau.util.model.value_error_pair import ValueErrorPair


def set_up_mocks() -> (
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
):
    legacy_factory_module_mock = MagicMock()
    legacy_factory_mock = MagicMock()
    legacy_factory_module_mock.LegacyFactory = legacy_factory_mock
    legacy_factory_mock.create_dataset = MagicMock()
    legacy_factory_mock.enrich_dataset = MagicMock()
    file_crawler_module_mock = MagicMock()
    file_crawler_mock = MagicMock()
    file_crawler_mock.crawl = MagicMock()
    file_crawler_module_mock.FileCrawler = file_crawler_mock
    file_crawler_module_mock.FileCrawler.return_value = (
        file_crawler_module_mock.FileCrawler
    )
    file_reader_module_mock = MagicMock()
    file_reader_mock = MagicMock()
    file_reader_module_mock.FileReader = file_reader_mock
    read_rows_mock = MagicMock()
    file_reader_mock.read_rows = read_rows_mock
    regex_module_mock = MagicMock()
    compile_regex_mock = MagicMock()
    regex_module_mock.compile = compile_regex_mock
    napytau_format_json_service_module_mock = MagicMock()
    napytau_format_json_service_mock = MagicMock()
    napytau_format_json_service_module_mock.NapytauFormatJsonService = (
        napytau_format_json_service_mock
    )
    napytau_factory_module_mock = MagicMock()
    napytau_factory_mock = MagicMock()
    napytau_factory_module_mock.NapyTauFactory = napytau_factory_mock
    napytau_factory_mock.create_dataset = MagicMock()
    napytau_factory_mock.enrich_dataset = MagicMock()

    return (
        legacy_factory_module_mock,
        file_crawler_module_mock,
        file_reader_module_mock,
        regex_module_mock,
        napytau_format_json_service_module_mock,
        napytau_factory_module_mock,
    )


class IngestUnitTest(unittest.TestCase):
    def test_instantiatesTheFileCrawlerWithAFitPatternIfNoFitFileIsProvided(self):
        """Instantiates the file crawler with a fit pattern if no fit file is provided."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            _,
        ) = set_up_mocks()
        regex_module_mock.compile = lambda x: x
        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_legacy_format_from_files,
            )

            import_legacy_format_from_files(PurePath("test_directory"))
            self.assertEqual(
                file_crawler_module_mock.FileCrawler.mock_calls[0].args[0],
                ["v_c", "distances.dat", "norm.fac", ".*.fit"],
            )

    def test_instantiatesTheFileCrawlerWithoutAFitPatternIfAFitFileIsProvided(self):
        """Instantiates the file crawler without a fit pattern if a fit file is provided."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            _,
        ) = set_up_mocks()
        regex_module_mock.compile = lambda x: x
        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_legacy_format_from_files,
            )

            import_legacy_format_from_files(
                PurePath("test_directory"), PurePath("test.fit")
            )
            self.assertEqual(
                file_crawler_module_mock.FileCrawler.mock_calls[0].args[0],
                ["v_c", "distances.dat", "norm.fac"],
            )

    @staticmethod
    def test_callsTheCrawlersCrawlMethodWithTheProvidedDirectoryPath():
        """Calls the crawler's crawl method with the provided directory path."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            _,
        ) = set_up_mocks()
        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_legacy_format_from_files,
            )

            import_legacy_format_from_files(PurePath("test_directory"))
            file_crawler_module_mock.FileCrawler.mock_calls[0].crawl("test_directory")

    def test_readsEveryFileReturnedByTheFileCrawler(self):
        """Reads every file returned by the file crawler."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            _,
        ) = set_up_mocks()
        file_crawler_module_mock.FileCrawler.return_value = (
            file_crawler_module_mock.FileCrawler
        )
        file_crawler_module_mock.FileCrawler.crawl.return_value = [
            LegacySetupFiles(
                PurePath("test_distances.dat"),
                PurePath("test_v_c"),
                PurePath("test_fit"),
                PurePath("test_norm.fac"),
            )
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_legacy_format_from_files,
            )

            import_legacy_format_from_files(PurePath("test_directory"))
            self.assertEqual(
                file_reader_module_mock.FileReader.read_rows.mock_calls[0].args[0],
                PurePath("test_v_c"),
            )
            self.assertEqual(
                file_reader_module_mock.FileReader.read_rows.mock_calls[1].args[0],
                PurePath("test_distances.dat"),
            )
            self.assertEqual(
                file_reader_module_mock.FileReader.read_rows.mock_calls[2].args[0],
                PurePath("test_fit"),
            )
            self.assertEqual(
                file_reader_module_mock.FileReader.read_rows.mock_calls[3].args[0],
                PurePath("test_norm.fac"),
            )

    def test_callsTheLegacyFactoryWithTheRawLegacyDataIfNoFitFileIsProvided(
        self,
    ):
        """Calls the legacy factory with the raw legacy data."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            _,
        ) = set_up_mocks()
        file_crawler_module_mock.FileCrawler.return_value = (
            file_crawler_module_mock.FileCrawler
        )
        file_crawler_module_mock.FileCrawler.crawl.return_value = [
            LegacySetupFiles(
                PurePath("test_distances.dat"),
                PurePath("test_v_c"),
                PurePath("test_fit"),
                PurePath("test_norm.fac"),
            )
        ]
        file_reader_module_mock.FileReader.read_rows.side_effect = [
            ["v_c_row"],
            ["distances.dat_row"],
            ["fit_row"],
            ["normalisation_row"],
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_legacy_format_from_files,
            )

            import_legacy_format_from_files(PurePath("test_directory"))
            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.create_dataset.mock_calls[0]
                .args[0]
                .velocity_rows,
                ["v_c_row"],
            )
            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.create_dataset.mock_calls[0]
                .args[0]
                .distance_rows,
                ["distances.dat_row"],
            )
            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.create_dataset.mock_calls[0]
                .args[0]
                .fit_rows,
                ["fit_row"],
            )
            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.create_dataset.mock_calls[0]
                .args[0]
                .normalisation_rows,
                ["normalisation_row"],
            )

    def test_callsTheLegacyFactoryWithTheRawLegacyDataIfAFitFileIsProvided(
        self,
    ):
        """Calls the legacy factory with the raw legacy data."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            _,
        ) = set_up_mocks()
        file_crawler_module_mock.FileCrawler.return_value = (
            file_crawler_module_mock.FileCrawler
        )
        file_crawler_module_mock.FileCrawler.crawl.return_value = [
            LegacySetupFiles(
                PurePath("test_distances.dat"),
                PurePath("test_v_c"),
                PurePath("test_fit"),
                PurePath("test_norm.fac"),
            )
        ]
        file_reader_module_mock.FileReader.read_rows.side_effect = [
            ["v_c_row"],
            ["distances.dat_row"],
            ["fit_row"],
            ["normalisation_row"],
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_legacy_format_from_files,
            )

            import_legacy_format_from_files(
                PurePath("test_directory"),
                PurePath("test.fit"),
            )
            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.create_dataset.mock_calls[0]
                .args[0]
                .velocity_rows,
                ["v_c_row"],
            )
            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.create_dataset.mock_calls[0]
                .args[0]
                .distance_rows,
                ["distances.dat_row"],
            )
            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.create_dataset.mock_calls[0]
                .args[0]
                .fit_rows,
                ["fit_row"],
            )
            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.create_dataset.mock_calls[0]
                .args[0]
                .normalisation_rows,
                ["normalisation_row"],
            )

    def test_usesTheFileReaderToReadTheRowsFromTheProvidedSetupFilePathWhenReadingLegacySetupData(
        self,
    ):
        """Uses the file reader to read the rows from the provided setup file path when reading Legacy setup data."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            _,
        ) = set_up_mocks()

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                read_legacy_setup_data_into_data_set,
            )

            dataset = DataSet(
                ValueErrorPair(RelativeVelocity(1), RelativeVelocity(0)),
                DatapointCollection([]),
            )

            read_legacy_setup_data_into_data_set(
                dataset,
                PurePath("test.napaset"),
            )

            self.assertEqual(
                file_reader_module_mock.FileReader.read_rows.mock_calls[0].args[0],
                PurePath("test.napaset"),
            )

    def test_callsTheLegacyFactoryToEnrichTheDatasetWithTheSetupDataReadByTheFileReader(
        self,
    ):
        """Calls the Legacy factory to enrich the dataset with the setup data read by the file reader."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            _,
        ) = set_up_mocks()

        file_reader_module_mock.FileReader.read_rows.return_value = ["row1", "row2"]

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                read_legacy_setup_data_into_data_set,
            )

            dataset = DataSet(
                ValueErrorPair(RelativeVelocity(1), RelativeVelocity(0)),
                DatapointCollection([]),
            )

            read_legacy_setup_data_into_data_set(
                dataset,
                PurePath("test.napaset"),
            )

            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.enrich_dataset.mock_calls[
                    0
                ].args[0],
                dataset,
            )
            self.assertEqual(
                legacy_factory_module_mock.LegacyFactory.enrich_dataset.mock_calls[
                    0
                ].args[1],
                RawLegacySetupData(["row1", "row2"]),
            )

    def test_configuresAndUsesTheFileCrawlerToSearchNapytauFormatFiles(self):
        """Configures and uses the file crawler to search Napytau format files."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            napytau_factory_module_mock,
        ) = set_up_mocks()
        regex_module_mock.compile = lambda x: x
        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
                "napytau.import_export.factory.napytau.napytau_factory": napytau_factory_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_napytau_format_from_files,
            )

            import_napytau_format_from_files(PurePath("test_directory"))

            self.assertEqual(
                file_crawler_module_mock.FileCrawler.mock_calls[0].args[0],
                [".*.napytau.json"],
            )

            self.assertEqual(
                file_crawler_module_mock.FileCrawler.crawl.mock_calls[0].args[0],
                PurePath("test_directory"),
            )

    def test_usesTheFileReaderToReadTheTextOfTheCrawledFiles(self):
        """Uses the file reader to read the text of the crawled files."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            napytau_factory_module_mock,
        ) = set_up_mocks()
        file_crawler_module_mock.FileCrawler.return_value = (
            file_crawler_module_mock.FileCrawler
        )
        file_crawler_module_mock.FileCrawler.crawl.return_value = [
            PurePath("test.napytau.json")
        ]

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
                "napytau.import_export.factory.napytau.napytau_factory": napytau_factory_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_napytau_format_from_files,
            )

            import_napytau_format_from_files(PurePath("test_directory"))

            self.assertEqual(
                file_reader_module_mock.FileReader.read_text.mock_calls[0].args[0],
                PurePath("test.napytau.json"),
            )

    def test_callsTheNapytauFormatJsonServiceToParseTheJsonData(self):
        """Calls the Napytau format JSON service to parse the JSON data."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            napytau_factory_module_mock,
        ) = set_up_mocks()
        file_crawler_module_mock.FileCrawler.return_value = (
            file_crawler_module_mock.FileCrawler
        )
        file_crawler_module_mock.FileCrawler.crawl.return_value = [
            PurePath("test.napytau.json")
        ]
        file_reader_module_mock.FileReader.read_text.return_value = "test_json_data"

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
                "napytau.import_export.factory.napytau.napytau_factory": napytau_factory_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_napytau_format_from_files,
            )

            import_napytau_format_from_files(PurePath("test_directory"))

            self.assertEqual(
                naptau_format_json_service_module_mock.NapytauFormatJsonService.parse_json_data.mock_calls[
                    0
                ].args[0],
                "test_json_data",
            )

    def test_createsADatasetFromTheParsedJsonData(self):
        """Creates a dataset from the parsed JSON data."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            napytau_factory_module_mock,
        ) = set_up_mocks()
        file_crawler_module_mock.FileCrawler.return_value = (
            file_crawler_module_mock.FileCrawler
        )
        file_crawler_module_mock.FileCrawler.crawl.return_value = [
            PurePath("test.napytau.json")
        ]
        file_reader_module_mock.FileReader.read_text.return_value = "test_json_data"
        naptau_format_json_service_module_mock.NapytauFormatJsonService.parse_json_data.return_value = {
            "relativeVelocity": 1,
            "relativeVelocityError": 0.1,
            "datapoints": [],
            "setups": ["setup1", "setup2"],
        }
        napytau_factory_module_mock.NapyTauFactory.create_dataset.return_value = (
            DataSet(
                ValueErrorPair(RelativeVelocity(1), RelativeVelocity(0.1)),
                DatapointCollection([]),
            )
        )

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
                "napytau.import_export.factory.napytau.napytau_factory": napytau_factory_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                import_napytau_format_from_files,
            )

            result = import_napytau_format_from_files(PurePath("test_directory"))

            self.assertEqual(
                napytau_factory_module_mock.NapyTauFactory.create_dataset.mock_calls[
                    0
                ].args[0],
                {
                    "relativeVelocity": 1,
                    "relativeVelocityError": 0.1,
                    "datapoints": [],
                    "setups": ["setup1", "setup2"],
                },
            )

            self.assertEqual(
                result[0][0].relative_velocity.value.velocity,
                1,
            )
            self.assertEqual(
                result[0][0].relative_velocity.error.velocity,
                0.1,
            )
            self.assertEqual(
                result[0][1],
                ["setup1", "setup2"],
            )

    def test_raisesAnExceptionIfTheSetupWithTheGivenNameIsNotFoundInTheProvidedRawSetups(
        self,
    ):
        """Raises an exception if the setup with the given name is not found in the provided raw setups."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            napytau_factory_module_mock,
        ) = set_up_mocks()
        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
                "napytau.import_export.factory.napytau.napytau_factory": napytau_factory_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                read_napytau_setup_data_into_data_set,
            )

            dataset = DataSet(
                ValueErrorPair(RelativeVelocity(1), RelativeVelocity(0.1)),
                DatapointCollection([]),
            )

            with self.assertRaises(Exception):
                read_napytau_setup_data_into_data_set(
                    dataset,
                    [{"name": "setup1"}],
                    "setup2",
                )

    def test_callsTheNapytauFactoryToEnrichTheGivenDatasetWithTheSetupData(self):
        """Calls the Napytau factory to enrich the given dataset with the setup data."""
        (
            legacy_factory_module_mock,
            file_crawler_module_mock,
            file_reader_module_mock,
            regex_module_mock,
            naptau_format_json_service_module_mock,
            napytau_factory_module_mock,
        ) = set_up_mocks()
        napytau_factory_module_mock.NapyTauFactory.enrich_dataset.return_value = (
            DataSet(
                ValueErrorPair(RelativeVelocity(1), RelativeVelocity(0.1)),
                DatapointCollection([]),
            )
        )

        with patch.dict(
            "sys.modules",
            {
                "napytau.import_export.factory.legacy.legacy_factory": legacy_factory_module_mock,
                "napytau.import_export.crawler.file_crawler": file_crawler_module_mock,
                "napytau.import_export.reader.file_reader": file_reader_module_mock,
                "re": regex_module_mock,
                "napytau.import_export.factory.napytau.json_service.napytau_format_json_service": naptau_format_json_service_module_mock,
                "napytau.import_export.factory.napytau.napytau_factory": napytau_factory_module_mock,
            },
        ):
            from napytau.import_export.import_export import (
                read_napytau_setup_data_into_data_set,
            )

            dataset = DataSet(
                ValueErrorPair(RelativeVelocity(1), RelativeVelocity(0.1)),
                DatapointCollection([]),
            )

            read_napytau_setup_data_into_data_set(
                dataset,
                [{"name": "setup1"}],
                "setup1",
            )

            self.assertEqual(
                napytau_factory_module_mock.NapyTauFactory.enrich_dataset.mock_calls[
                    0
                ].args[0],
                dataset,
            )
            self.assertEqual(
                napytau_factory_module_mock.NapyTauFactory.enrich_dataset.mock_calls[
                    0
                ].args[1],
                {"name": "setup1"},
            )


if __name__ == "__main__":
    unittest.main()

from pathlib import PurePath
from re import compile as compile_regex
from typing import Optional, List, Tuple

from napytau.import_export.factory.legacy.legacy_factory import (
    LegacyFactory,
)
from napytau.import_export.factory.legacy.raw_legacy_data import RawLegacyData
from napytau.import_export.crawler.file_crawler import FileCrawler
from napytau.import_export.crawler.legacy_setup_files import LegacySetupFiles
from napytau.import_export.factory.legacy.raw_legacy_setup_data import (
    RawLegacySetupData,
)
from napytau.import_export.factory.napytau.json_service.napytau_format_json_service import (  # noqa E501
    NapytauFormatJsonService,
)
from napytau.import_export.factory.napytau.napytau_factory import NapyTauFactory
from napytau.import_export.import_export_error import ImportExportError
from napytau.import_export.model.dataset import DataSet
from napytau.import_export.reader.file_reader import FileReader

IMPORT_FORMAT_LEGACY = "legacy"
IMPORT_FORMAT_NAPYTAU = "napytau"

IMPORT_FORMATS = [IMPORT_FORMAT_LEGACY, IMPORT_FORMAT_NAPYTAU]


def import_legacy_format_from_files(
    directory_path: PurePath, fit_file_path: Optional[PurePath] = None
) -> List[DataSet]:
    """
    Ingests a dataset from the Legacy format. The directory path will be
    recursively searched for the following files:
    - v_c
    - distances.dat
    - norm.fac
    - optional: *.fit

    if the fit_file_path is provided, the fit file will be
    read from this path instead of the directory.
    """

    file_crawler = _configure_file_crawler_for_legacy_format(fit_file_path)

    setup_file_bundles: List[LegacySetupFiles] = file_crawler.crawl(directory_path)

    return list(
        map(
            lambda setup_files: LegacyFactory.create_dataset(
                RawLegacyData(
                    FileReader.read_rows(setup_files.velocity_file),
                    FileReader.read_rows(setup_files.distances_file),
                    FileReader.read_rows(setup_files.fit_file),
                    FileReader.read_rows(setup_files.normalisation_file),
                )
            ),
            setup_file_bundles,
        )
    )


def _configure_file_crawler_for_legacy_format(
    fit_file_path: Optional[PurePath],
) -> FileCrawler:
    if fit_file_path:
        file_crawler = FileCrawler(
            [
                compile_regex("v_c"),
                compile_regex("distances.dat"),
                compile_regex("norm.fac"),
            ],
            lambda files: LegacySetupFiles.create_from_file_names(
                files + [fit_file_path]
            ),
        )

    else:
        file_crawler = FileCrawler(
            [
                compile_regex("v_c"),
                compile_regex("distances.dat"),
                compile_regex("norm.fac"),
                compile_regex(".*.fit"),
            ],
            lambda files: LegacySetupFiles.create_from_file_names(files),
        )
    return file_crawler


def read_legacy_setup_data_into_data_set(
    dataset: DataSet, setup_file_path: PurePath
) -> DataSet:
    """
    Reads the setup data from the provided file path and adds it to the provided dataset
    """

    setup_data = FileReader.read_rows(setup_file_path)

    return LegacyFactory.enrich_dataset(dataset, RawLegacySetupData(setup_data))


def import_napytau_format_from_files(
    directory_path: PurePath,
) -> List[Tuple[DataSet, List[dict]]]:
    """
    Ingests a dataset from the NapyTau format. The directory path will be
    recursively searched for the following files:
    - napytau.json

    :param directory_path: The directory path to search for the .napytau.json files

    :return: A list of datasets and their corresponding raw setup data
    """

    file_crawler = FileCrawler(
        [compile_regex(".*.napytau.json")],
        lambda files: files[0],
    )

    napytau_file_paths = file_crawler.crawl(directory_path)

    return list(
        map(
            lambda napytau_file_path: _map_raw_napytau_data(napytau_file_path),
            napytau_file_paths,
        )
    )


def _map_raw_napytau_data(napytau_file_path: PurePath) -> Tuple[DataSet, List[dict]]:
    json_data = NapytauFormatJsonService.parse_json_data(
        FileReader.read_text(napytau_file_path)
    )

    return (
        NapyTauFactory.create_dataset(json_data),
        json_data["setups"],
    )


def read_napytau_setup_data_into_data_set(
    dataset: DataSet, raw_setups_data: List[dict], setup_name: str
) -> DataSet:
    """
    Reads the setup data from the provided file path and adds it to the provided dataset

    :param dataset: The dataset to enrich
    :param raw_setups_data: The raw json data of the datasets associated setups
    :param setup_name: The name of the setup to add to the dataset

    :return: The enriched dataset
    """

    raw_setup_data = next(
        (setup for setup in raw_setups_data if setup["name"] == setup_name),
        None,
    )

    if not raw_setup_data:
        raise ImportExportError(
            f"Setup with name {setup_name} not found in the provided data"
        )

    return NapyTauFactory.enrich_dataset(
        dataset,
        raw_setup_data,
    )

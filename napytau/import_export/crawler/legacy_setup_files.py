from __future__ import annotations

from pathlib import PurePath
from typing import List

from napytau.import_export.import_export_error import ImportExportError


class LegacySetupFiles:
    distances_file: PurePath
    velocity_file: PurePath
    fit_file: PurePath
    normalisation_file: PurePath

    def __init__(
        self,
        distances_file: PurePath,
        velocity_file: PurePath,
        fit_file: PurePath,
        normalisation_file: PurePath,
    ):
        """Use the static constructor method instead"""
        self.distances_file = distances_file
        self.velocity_file = velocity_file
        self.fit_file = fit_file
        self.normalisation_file = normalisation_file

    @staticmethod
    def create_from_file_names(file_paths: List[PurePath]) -> LegacySetupFiles:
        distances_file: PurePath
        velocity_file: PurePath
        fit_file: PurePath
        normalisation_file: PurePath
        missing_files = []

        try:
            distances_file = next(
                file for file in file_paths if "distances.dat" in file.name
            )
        except StopIteration:
            missing_files.append("distances.dat")

        try:
            velocity_file = next(file for file in file_paths if "v_c" in file.name)
        except StopIteration:
            missing_files.append("v_c")

        try:
            fit_file = next(file for file in file_paths if "fit" in file.name)
        except StopIteration:
            missing_files.append("fit")

        try:
            normalisation_file = next(
                file for file in file_paths if "norm.fac" in file.name
            )
        except StopIteration:
            missing_files.append("norm.fac")

        if len(missing_files) > 0:
            raise ImportExportError(
                f"Could not find all necessary files in the provided list of file names"
                f". Missing files: {missing_files}"
            )

        return LegacySetupFiles(
            distances_file, velocity_file, fit_file, normalisation_file
        )

from dataclasses import dataclass
from typing import List


@dataclass
class RawLegacyData:
    velocity_rows: List[str]
    distance_rows: List[str]
    fit_rows: List[str]
    normalisation_rows: List[str]

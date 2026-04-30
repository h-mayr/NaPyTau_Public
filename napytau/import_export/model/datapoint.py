from dataclasses import dataclass
from typing import Optional, Tuple

from napytau.util.model.value_error_pair import ValueErrorPair


@dataclass
class Datapoint:
    """
    A class to represent a single datapoint in a dataset.
    Distance acts as a key, identifying the datapoint, therefore it is required.
    All other attributes are optional and can be set later.

    As this class sits at the core of the entire system, it is important to take care
    when modifying it. Any changes to this class will have a ripple effect on the entire
    system.
    """

    distance: ValueErrorPair[float]
    normalisation: Optional[ValueErrorPair[float]] = None
    shifted_intensity: Optional[ValueErrorPair[float]] = None
    unshifted_intensity: Optional[ValueErrorPair[float]] = None
    feeding_shifted_intensity: Optional[ValueErrorPair[float]] = None
    feeding_unshifted_intensity: Optional[ValueErrorPair[float]] = None
    tau: Optional[ValueErrorPair[float]] = None
    active: bool = True

    def get_distance(self) -> ValueErrorPair[float]:
        return self.distance

    def set_distance(self, distance: ValueErrorPair[float]) -> None:
        self.distance = distance

    def get_normalisation(self) -> ValueErrorPair[float]:
        if self.normalisation is None:
            raise ValueError("Normalisation was accessed before initialization.")

        return self.normalisation

    def set_normalisation(self, normalisation: ValueErrorPair[float]) -> None:
        self.normalisation = normalisation

    def get_intensity(self) -> Tuple[ValueErrorPair[float], ValueErrorPair[float]]:
        if self.shifted_intensity is None or self.unshifted_intensity is None:
            raise ValueError("Intensity was accessed before initialization.")

        return (
            self.shifted_intensity,
            self.unshifted_intensity,
        )

    def set_intensity(
        self,
        shifted_intensity: ValueErrorPair[float],
        unshifted_intensity: ValueErrorPair[float],
    ) -> None:
        self.shifted_intensity = shifted_intensity
        self.unshifted_intensity = unshifted_intensity

    def get_feeding_intensity(
        self,
    ) -> Tuple[Optional[ValueErrorPair[float]], Optional[ValueErrorPair[float]]]:
        return (
            self.feeding_shifted_intensity,
            self.feeding_unshifted_intensity,
        )

    def set_feeding_intensity(
        self,
        feeding_shifted_intensity: ValueErrorPair[float],
        feeding_unshifted_intensity: ValueErrorPair[float],
    ) -> None:
        self.feeding_shifted_intensity = feeding_shifted_intensity
        self.feeding_unshifted_intensity = feeding_unshifted_intensity

    def get_tau(self) -> ValueErrorPair[float]:
        if self.tau is None:
            raise ValueError("Tau was accessed before initialization.")

        return self.tau

    def set_tau(self, tau: ValueErrorPair[float]) -> None:
        self.tau = tau

    def is_active(self) -> bool:
        return self.active

    def set_active(self, active: bool) -> None:
        self.active = active

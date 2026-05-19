from dataclasses import dataclass
from typing import Optional


@dataclass
class Polynomial:
    """
    A class to represent a polynomial.
    A polynomial is a mathematical expression consisting of variables and coefficients.
    """

    coefficients: list[float]
    knots: Optional[list[float]] = None
    degree: Optional[int] = None

    def get_coefficients(self) -> list[float]:
        return self.coefficients

    def set_coefficients(self, coefficients: list[float]) -> None:
        self.coefficients = coefficients

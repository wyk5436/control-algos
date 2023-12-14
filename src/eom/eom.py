import abc
import numbers
import numpy as np

class EOM(abc.ABC):
    """EOM, Parent class for all Equation of Motion classes."""

    __author__ = "bryantzhou"

    @abc.abstractmethod
    def evaluate(self, stepsize: numbers.Real, time: numbers.Real, states: np.ndarray) -> np.ndarray:
        pass

    
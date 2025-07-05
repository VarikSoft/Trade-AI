from abc import ABC, abstractmethod
import pandas as pd

class StrategyBase(ABC):
    """
    Base interface for any trading strategy.
    """

    @abstractmethod
    def __init__(self, **params):
        """
        Accepts a dictionary of strategy parameters.
        """
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Takes as input a DataFrame with OHLCV data and
        returns a pd.Series of {-1, 0, +1} ('sell', 'hold', 'buy')
        with the same index as df.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """
        Unique strategy name or key, e.g., "ma_crossover".
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """
        Returns the current strategy parameters for logging/GA.
        """
        ...
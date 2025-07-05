import pandas as pd
from strategy_base import StrategyBase

class MACrossover(StrategyBase):
    def __init__(self, fast: int = 10, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def get_name(self) -> str:
        return "ma_crossover"

    def get_params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Signal +1 when the fast MA crosses the slow MA from below;
        -1 when it crosses from above; 0 otherwise.
        """
        ma_fast = df['Close'].rolling(self.fast).mean()
        ma_slow = df['Close'].rolling(self.slow).mean()
        cross_up   = (ma_fast.shift(1) < ma_slow.shift(1)) & (ma_fast >= ma_slow)
        cross_down = (ma_fast.shift(1) > ma_slow.shift(1)) & (ma_fast <= ma_slow)

        signals = pd.Series(0, index=df.index)
        signals[cross_up]   =  1
        signals[cross_down] = -1
        return signals
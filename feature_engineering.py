import pandas as pd
import numpy as np

class FEDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return FEDataFrame
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, pd.Series):
            if isinstance(key, tuple):
                result = result.rename(key[-1])
            elif isinstance(key, str) and key.endswith('_Signal'):
                result = result.rename(key.replace('_Signal',''))
        return result

class FeatureEngineer:
    def __init__(self, data_dict):
        self.data = {sym: FEDataFrame(df) for sym, df in data_dict.items()}
    def sma(self, window):
        for sym, df in self.data.items():
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
        return self
    def ema(self, span):
        for sym, df in self.data.items():
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        return self
    def rsi(self, window):
        for sym, df in self.data.items():
            delta = df['Close'].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ma_up = up.rolling(window).mean()
            ma_down = down.rolling(window).mean()
            rs = ma_up / ma_down
            df[f'RSI_{window}'] = 100 - 100 / (1 + rs)
        return self
    def macd(self, fast=12, slow=26, signal=9):
        for sym, df in self.data.items():
            ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            df['MACD'] = macd_line
            df['MACD_Signal'] = signal_line
        return self
    def get_data(self):
        combined = pd.concat(self.data, axis=1)
        return FEDataFrame(combined)
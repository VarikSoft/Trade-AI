import os
import pandas as pd
import yfinance as yf

class DataLoader:
    def __init__(self, symbols, start_date, end_date, interval='1d', source='yfinance', csv_paths=None):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.source = source
        self.csv_paths = csv_paths or {}
        self._cache = {}

    def fetch(self):
        result = {}
        for sym in self.symbols:
            if sym in self._cache:
                df = self._cache[sym]
            elif self.source == 'csv' and sym in self.csv_paths:
                path = self.csv_paths[sym]
                df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
                df.index.name = None
                freq = pd.infer_freq(df.index)
                df.index.freq = freq
                self._cache[sym] = df
            else:
                df = yf.download(sym, start=self.start_date, end=self.end_date, interval=self.interval)
                self._cache[sym] = df
            result[sym] = df
        return result

    def to_dataframe(self):
        data = self.fetch()
        df = pd.concat(data, axis=1, names=['Symbol', 'Field'])
        df.columns = [f"{sym}_{field}" for sym, field in df.columns]
        return df

    def resample(self, data, rule):
        agg = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Adj Close': 'last',
            'Volume': 'sum'
        }
        return {
            sym: df.resample(rule, closed='right', label='right').agg(agg).dropna()
            for sym, df in data.items()
        }
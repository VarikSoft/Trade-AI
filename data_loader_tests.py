import os
import tempfile
import pandas as pd
import pytest
from data_loader import DataLoader
import yfinance as yf

class DummyDF(pd.DataFrame):
    @property
    def _constructor(self):
        return DummyDF

@pytest.fixture
def dummy_df():
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    return pd.DataFrame({
        'Open': [1,2,3],
        'High': [2,3,4],
        'Low': [0,1,2],
        'Close': [1.5,2.5,3.5],
        'Adj Close': [1.5,2.5,3.5],
        'Volume': [100,200,300]
    }, index=dates)

def test_fetch_yfinance(monkeypatch, dummy_df):
    monkeypatch.setattr(yf, 'download', lambda sym, start, end, interval: dummy_df)
    dl = DataLoader(['AAA'], '2020-01-01', '2020-01-04')
    data = dl.fetch()
    assert 'AAA' in data
    pd.testing.assert_frame_equal(data['AAA'], dummy_df)

def test_fetch_csv(tmp_path, dummy_df):
    p = tmp_path / "AAA.csv"
    df = dummy_df.copy()
    df = df.reset_index().rename(columns={'index':'Date'})
    df.to_csv(p, index=False)
    dl = DataLoader(['AAA'], '2020-01-01', '2020-01-04', source='csv', csv_paths={'AAA': str(p)})
    data = dl.fetch()
    assert 'AAA' in data
    pd.testing.assert_frame_equal(data['AAA'], dummy_df)

def test_to_dataframe(monkeypatch, dummy_df):
    monkeypatch.setattr(DataLoader, 'fetch', lambda self: {'A': dummy_df, 'B': dummy_df})
    dl = DataLoader(['A','B'], '2020-01-01', '2020-01-04')
    df = dl.to_dataframe()
    expected_cols = [f"{sym}_{field}" for sym in ['A','B'] for field in dummy_df.columns]
    assert list(df.columns) == expected_cols
    assert len(df) == len(dummy_df)

def test_resample(dummy_df):
    data = {'A': dummy_df}
    dl = DataLoader(['A'], '2020-01-01', '2020-01-04')
    out = dl.resample(data, '2D')
    assert 'A' in out
    df2 = out['A']
    assert list(df2.index) == [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-03')]
    assert df2.loc['2020-01-01','High'] == 2
    assert df2.loc['2020-01-03','Low'] == 1
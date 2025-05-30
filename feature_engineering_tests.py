import pandas as pd
import numpy as np
import pytest
from feature_engineering import FeatureEngineer

@pytest.fixture
def dummy_close_df():
    dates = pd.date_range('2021-01-01', periods=5, freq='D')
    return pd.DataFrame({'Close': [1, 2, 3, 4, 5]}, index=dates)

def test_sma(dummy_close_df):
    fe = FeatureEngineer({'X': dummy_close_df.copy()})
    fe.sma(3)
    result = fe.data['X']['SMA_3']
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0],
                         index=dummy_close_df.index,
                         name='SMA_3')
    pd.testing.assert_series_equal(result, expected)

def test_ema(dummy_close_df):
    fe = FeatureEngineer({'X': dummy_close_df.copy()})
    fe.ema(3)
    result = fe.data['X']['EMA_3']
    # alpha = 2/(3+1) = 0.5
    expected_values = [1.0,
                       1.5,       # 2*0.5 + 1*0.5
                       2.25,      # 3*0.5 + 1.5*0.5
                       3.125,     # 4*0.5 + 2.25*0.5
                       4.0625]    # 5*0.5 + 3.125*0.5
    expected = pd.Series(expected_values,
                         index=dummy_close_df.index,
                         name='EMA_3')
    pd.testing.assert_series_equal(result, expected)

def test_rsi_simple():
    dates = pd.date_range('2021-01-01', periods=3, freq='D')
    df = pd.DataFrame({'Close': [1, 0, 1]}, index=dates)
    fe = FeatureEngineer({'A': df.copy()})
    fe.rsi(2)
    rs = fe.data['A']['RSI_2']
    assert np.isnan(rs.iloc[0])
    assert np.isnan(rs.iloc[1])
    assert pytest.approx(50.0) == rs.iloc[2]

def test_macd():
    dates = pd.date_range('2021-01-01', periods=5, freq='D')
    df = pd.DataFrame({'Close': [1, 2, 3, 4, 5]}, index=dates)
    fe = FeatureEngineer({'SYM': df.copy()})
    fe.macd(fast=2, slow=3, signal=2)
    out = fe.data['SYM']
    macd_line = out['MACD']
    signal = out['MACD_Signal']
    expected_signal = macd_line.ewm(span=2, adjust=False).mean()
    pd.testing.assert_series_equal(signal, expected_signal)

def test_get_data(dummy_close_df):
    df1 = dummy_close_df.copy()
    df2 = dummy_close_df.copy() * 2
    fe = FeatureEngineer({'A': df1, 'B': df2})
    combined = fe.get_data()
    assert ('A', 'Close') in combined.columns
    assert ('B', 'Close') in combined.columns
    pd.testing.assert_series_equal(combined[('A', 'Close')], df1['Close'])
    pd.testing.assert_series_equal(combined[('B', 'Close')], df2['Close'])

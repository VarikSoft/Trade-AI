import os
import sys
import argparse
import pandas as pd
import numpy as np

# Import your strategy code
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(PROJECT_ROOT, 'Trade-Ai'))

from strategy_loader import discover_strategies
from strategy_base import StrategyBase

def annualization_factor(freq: str) -> float:
    """
    Approximate annualization factors for different data frequencies.
    """
    mapping = {
        'D': 252,           # daily bars
        'B': 252,           # business days
        'H': 252 * 6.5,     # 6.5 trading hours per day
        'T': 252 * 6.5 * 4  # 15-minute bars
    }
    return mapping.get(freq[0], 252)

def compute_performance(df: pd.DataFrame,
                        signal_col: str = 'signal',
                        price_col: str = 'Close',
                        capital: float = 1.0) -> dict:
    """
    Takes a DataFrame with signal and closing price columns.
    Returns a dict with performance metrics and the equity series.
    """
    # 1) Calculate period returns of the price
    df['price_ret'] = df[price_col].pct_change().fillna(0)

    # 2) Strategy return: previous signal * price return
    df['strat_ret'] = df[signal_col].shift(1).fillna(0) * df['price_ret']

    # 3) Cumulative equity curve
    df['equity'] = (1 + df['strat_ret']).cumprod() * capital

    # 4) Metrics
    total_ret = df['equity'].iloc[-1] / capital - 1
    mean_ret = df['strat_ret'].mean()
    std_ret  = df['strat_ret'].std()

    # Determine data frequency and annualize
    freq = pd.infer_freq(df.index) or 'D'
    ann_factor = annualization_factor(freq)

    ann_ret = (1 + mean_ret) ** ann_factor - 1
    ann_vol = std_ret * np.sqrt(ann_factor)
    sharpe  = (mean_ret / std_ret) * np.sqrt(ann_factor) if std_ret != 0 else np.nan

    # Maximum drawdown
    running_max = df['equity'].cummax()
    drawdown = df['equity'] / running_max - 1
    max_dd = drawdown.min()

    return {
        'total_return': total_ret,
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'equity_series': df['equity'],
        'returns_series': df['strat_ret']
    }

def main():
    parser = argparse.ArgumentParser(
        description="Backtest a single strategy on an OHLCV+features CSV"
    )
    parser.add_argument('-d', '--data', required=True,
                        help="CSV file with date index and OHLCV plus feature columns")
    parser.add_argument('-s', '--strategy', required=True,
                        help="Strategy key (get_name()), e.g.: ma_crossover")
    parser.add_argument('-p', '--params', nargs='*', default=[],
                        help="Strategy parameters as key=value pairs, e.g. fast=10 slow=50")
    parser.add_argument('-c', '--capital', type=float, default=1.0,
                        help="Starting capital (default: 1.0)")
    parser.add_argument('-o', '--output', default='equity.csv',
                        help="Path to save the equity curve CSV")
    args = parser.parse_args()

    # 1) Load the data
    df = pd.read_csv(args.data, index_col=0, parse_dates=True, infer_datetime_format=True)
    df = df.sort_index()

    # 2) Discover the strategy class
    strat_dir = os.path.join(PROJECT_ROOT, 'strategies')
    strat_classes = discover_strategies(strat_dir)
    strat_cls = None
    for cls in strat_classes:
        inst = cls()  # instantiate without params to check its name
        if inst.get_name() == args.strategy:
            strat_cls = cls
            break
    if strat_cls is None:
        print(f"❌ Strategy '{args.strategy}' not found in strategies/")
        print("Available strategies:", [cls().get_name() for cls in strat_classes])
        sys.exit(1)

    # 3) Parse parameters
    init_args = {}
    for kv in args.params:
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        try:
            init_args[k] = int(v)
        except ValueError:
            try:
                init_args[k] = float(v)
            except ValueError:
                init_args[k] = v

    # 4) Instantiate the strategy
    strat: StrategyBase = strat_cls(**init_args)

    # 5) Generate signals
    signals = strat.generate_signals(df)
    df['signal'] = signals

    # 6) Run backtest
    perf = compute_performance(df, signal_col='signal', price_col='Close', capital=args.capital)

    # 7) Print results
    print("\n=== Backtest results for strategy:", strat.get_name(), "===\n")
    print(f"Total return:       {perf['total_return'] * 100:.2f}%")
    print(f"Annualized return:  {perf['annual_return'] * 100:.2f}%")
    print(f"Annual volatility:  {perf['annual_volatility'] * 100:.2f}%")
    print(f"Sharpe ratio:       {perf['sharpe_ratio']:.2f}")
    print(f"Max drawdown:       {perf['max_drawdown'] * 100:.2f}%")
    print("\nEquity curve saved to:", args.output)

    # 8) Save the equity curve
    out_df = pd.DataFrame({
        'equity': perf['equity_series']
    })
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    out_df.to_csv(args.output)

if __name__ == "__main__":
    main()
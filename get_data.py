import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

MAX_LOOKBACK = {
    '15m': timedelta(days=60),
    '1h':  timedelta(days=730),
    '1d':  None,
    '1wk': None,
}

INTERVALS = {
    '15m': '15m',
    '1h':  '60m',
    '1d':  '1d',
    '1wk': '1wk'
}

def fetch_interval_chunks(ticker: str, interval_key: str, start: datetime, end: datetime) -> pd.DataFrame:
    max_delta = MAX_LOOKBACK[interval_key]
    yf_interval = INTERVALS[interval_key]

    if max_delta is None:
        return yf.download(
            tickers=ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=yf_interval,
            auto_adjust=True,
            progress=False
        )

    dfs = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + max_delta, end)
        print(f"  Downloading {ticker} {interval_key}: {window_start.date()} → {window_end.date()}")
        try:
            df_chunk = yf.download(
                tickers=ticker,
                start=window_start.strftime("%Y-%m-%d"),
                end=window_end.strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,
                progress=False
            )
            if not df_chunk.empty:
                dfs.append(df_chunk)
        except Exception as e:
            print(f"   ‼ Error: {window_start.date()}–{window_end.date()}: {e}")
        window_start = window_end

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs)
    df = df[~df.index.duplicated(keep='first')]
    return df

def fetch_and_save(ticker: str, interval_key: str, start: str, end: str, out_dir: str):
    start_dt = datetime.fromisoformat(start)
    end_dt   = datetime.fromisoformat(end)

    print(f"→ {ticker} @ {interval_key} from {start} to {end}")
    df = fetch_interval_chunks(ticker, interval_key, start_dt, end_dt)
    if df.empty:
        print(f"‼️ No data for {ticker} at {interval_key}")
        return

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{ticker}_{interval_key}.csv")
    df.to_csv(fname)
    print(f"Saved to {fname}")

def main():
    parser = argparse.ArgumentParser(description="Fetch historical data for trading AI")
    parser.add_argument('-t', '--tickers', nargs='+', required=True,
                        help="Tickers list, e.g.: AAPL MSFT GOOGL")
    parser.add_argument('-s', '--start',  default="2015-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument('-e', '--end',    default=datetime.today().strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD")
    parser.add_argument('-o', '--outdir', default="data",
                        help="Output folder for CSV files")
    args = parser.parse_args()

    for ticker in args.tickers:
        for interval_key in INTERVALS:
            fetch_and_save(ticker, interval_key, args.start, args.end, args.outdir)

if __name__ == "__main__":
    main()
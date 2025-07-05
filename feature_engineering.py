import os
import argparse
import pandas as pd
import numpy as np

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(close: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9):
    ema_fast    = close.ewm(span=fast, adjust=False).mean()
    ema_slow    = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist   = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def compute_bollinger(close: pd.Series,
                      period: int = 20,
                      std_factor: int = 2):
    sma   = close.rolling(window=period).mean()
    std   = close.rolling(window=period).std()
    upper = sma + std_factor * std
    lower = sma - std_factor * std
    return sma, upper, lower

def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()

def process_file(input_path: str, output_path: str):
    # 1) Загрузка и разбор дат
    df = pd.read_csv(
        input_path,
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True
    )
    df = df.sort_index()

    # 2) Приводим колонки к числовому типу (строки → float/int)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3) Убираем строки с NaN в базовых колонках
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    # 4) Расчёт индикаторов
    df['rsi_14'] = compute_rsi(df['Close'], period=14)

    macd_line, macd_signal, macd_hist = compute_macd(df['Close'])
    df['macd_line']   = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist']   = macd_hist

    sma, upper, lower = compute_bollinger(df['Close'], period=20, std_factor=2)
    df['bb_sma_20']   = sma
    df['bb_upper_20'] = upper
    df['bb_lower_20'] = lower

    df['obv'] = compute_obv(df)

    # 5) Удаляем начальные строки с NaN от индикаторов
    df.dropna(inplace=True)

    # 6) Сохраняем результат
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"✔ Processed {os.path.basename(input_path)} → {os.path.basename(output_path)}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute technical indicators for OHLCV CSV files"
    )
    parser.add_argument(
        '-i', '--input-dir', required=True,
        help="Папка с исходными CSV (каждый с колонками Open,High,Low,Close,Volume)"
    )
    parser.add_argument(
        '-o', '--output-dir', default='features',
        help="Папка для сохранения CSV с добавленными признаками"
    )
    args = parser.parse_args()

    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith('.csv'):
            continue
        in_path   = os.path.join(args.input_dir, fname)
        out_fname = fname.replace('.csv', '_features.csv')
        out_path  = os.path.join(args.output_dir, out_fname)
        process_file(in_path, out_path)

if __name__ == "__main__":
    main()
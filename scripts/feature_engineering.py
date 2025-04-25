import pandas as pd
import pandas_ta as ta 
import os

os.makedirs("outputs/processed_data", exist_ok=True)
files = ['BTC_USD.csv', "EUR_USD.csv", 'SPY.csv']
for file in files:
    symbol = file.replace(".csv", '')
    print(f"Processing: {symbol}")
    df = pd.read_csv(f'data/{file}', parse_dates = ['Date'])
    df.set_index("Date", inplace=True)
    df.dropna(inplace=True)

    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD_Line'], df['MACD_Signal'], _ = ta.macd(df['Close'])
    bb = ta.bbands9df['Close']
    df = df.join(bb)

    df["SMA_20"] = ta.sma(df['Close'], length=20)
    df["EMA_20"] = ta.ema(df["Close"], length=20)
    df["OBV"] = ta.obv(df["Close"], df["Volume"])
    df.dropna(inplace=True)

    out_path = f'outputs/processed_data/{symbol}_processed.csv'
    df.to_csv(out_path)
    print(f"Saved: {out_path}")

print("All datasets feature-engineered")
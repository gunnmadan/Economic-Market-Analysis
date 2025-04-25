import yfinance as yf 
import os 

tickers = {
    "BTC_USD" : "BTC-USD",
    "EUR_USD": "EURUSD=X",
    "SPY" : "SPY"   
}
os.makedirs("data", exist_ok=True)

for name, ticker in tickers.items():
    df = yf.download(ticker, start='2018-01-01', end='2024-12-31', interval='1d')
    df.to_csv(f'data/{name}.csv')
    print(f"Saved: data/{name}.csv")

print("All datasets downloaded")

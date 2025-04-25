import pandas as pd 
import os 

input_dir = 'outputs/processed_data'
output_dir = 'outputs/labeled_data'
os.makedirs(output_dir, exist_ok=True)

files = ['BTC_USD_processed.csv', 'EUR_USD_processed.csv', 'SPY_processed.csv']

for file in files: 
    df = pd.read_csv(f"{input_dir}/{file}", parse_dates=["Price"])
    df.set_index("Price", inplace=True)
    df["Target_1d"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["Target_5d"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    output_path = f"{output_dir}/{file}"
    df.to_csv(output_path)
    print(f"Labeled data saved to: {output_path}")

print("Labels added to all datasets")
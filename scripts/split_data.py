import pandas as pd 
import os 

input_dir = 'outputs/labeled_data'
output_dir = 'outputs/splits'
os.makedirs(output_dir, exist_ok=True)

files = ['BTC_USD_processed.csv', 'EUR_USD_processed.csv', 'SPY_processed.csv']

for file in files:
    df = pd.read_csv(f"{input_dir}/{file}", parse_dates=["Price"])
    df.set_index("Price", inplace=True)
    
    train_end = int(0.8 * len(df))
    val_end = int(0.9 * len(df))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    base = file.replace("_processed.csv", "")
    train.to_csv(f"{output_dir}/{base}_train.csv")
    val.to_csv(f"{output_dir}/{base}_val.csv")
    test.to_csv(f"{output_dir}_test.csv")

    print(f"Split complete for {base}")

print("All datasets split into train, val, test")
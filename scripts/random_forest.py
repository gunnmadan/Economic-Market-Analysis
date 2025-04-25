import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter 
import os 

input_path = 'outputs/splits/BTC_USD'
train = pd.read_csv(f"{input_path}_train.csv", parse_dates=['Price'], index_col='Price')
val = pd.read_csv(f"{input_path}_val.csv", parse_dates=["Price"], index_col='Price')
features = ['RSI', "MACD_12_26_9", "MACDs_12_26_9", "BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0", "SMA_20", "EMA_20", "OBV"]
target = "Target_1d"

X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
print(f"Random Forest Accuracy (BTC/USD, 1-Day): {acc:.4f}")

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("Prediction distribution:", Counter(y_pred))

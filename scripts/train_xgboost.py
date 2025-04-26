import pandas as pd 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

input_path = 'outputs/splits/BTC_USD'
train = pd.read_csv(f"{input_path}_train.csv", parse_dates = ["Price"], index_col='Price')
val = pd.read_csv(f"{input_path}_val.csv", parse_dates=['Price'], index_col='Price')

features = [
    'RSI', "MACD_12_26_9", "MACDs_12_26_9",
    "BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0",
    "SMA_20", "EMA_20", "OBV"
]
target = "Target_1d"

X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"\n XGBoost Accuracy (BTC/USD, 1-Day): {acc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
top_features = [features[i] for i in sorted_idx[:10]]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[sorted_idx[:10]], y=top_features, palette='coolwarm')
plt.title("Top 10 Feature Importances (XGBoost - BTC/USD)")
plt.xlabel("Importance")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(12, 5))
plt.plot(y_val.values, label="Actual", linestyle="--", marker='o', markersize=3)
plt.plot(y_pred, label="Predicted", linestyle="-", marker='x', markersize=3)
plt.title("Actual vs Predicted Classes (Validation Set - XGBoost)")
plt.xlabel("Samples")
plt.ylabel("Class (0 = Down, 1 = Up)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

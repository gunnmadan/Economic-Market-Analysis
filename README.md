# Using Machine Learning to Analyze Economic Market Prices with Trading Technical Analysis
## 🤖 ML Spring 2025 Project Overview

This project uses machine learning to predict short-term price trends (1-day and 5-day) in three markets: Bitcoin (BTC/USD), Euro to U.S. Dollar (EUR/USD), and the SPY ETF. The aim is to automate trading insights that are usually manual, using models trained on historical price and volume data.

We compared three models:  
- **Random Forest** (via scikit-learn)  
- **XGBoost** (for performance and scale)  
- **LSTM** (built with PyTorch for sequential pattern recognition)

Technical indicators like RSI, MACD, and OBV were engineered from the data to improve prediction accuracy. We focused on creating a system that's data-driven, scalable, and reproducible.

Developer Docs:
* https://scikit-learn.org/stable/user_guide.html 
* https://pandas.pydata.org/docs/ 
* https://numpy.org/doc/stable/ 
* https://xgboost.readthedocs.io/

This blog post from NVIDIA on LSTMs was a huge help:
https://developer.nvidia.com/discover/lstm#:~:text=A%20Long%20short%2Dterm%20memory,cycles%20through%20the%20feedback%20loops

---

## 💡 Key Contributions

- Developed a multi-model system to predict trends in crypto, forex, and equities  
- Built and compared Random Forest, XGBoost, and LSTM models  
- Engineered features from technical indicators  
- Preprocessed and normalized raw OHLCV data  
- Evaluated models using metrics like Directional Accuracy, RMSE, and MAE  

---

## 🛠️ Technical Summary

We pulled historical daily data from Yahoo Finance (BTC, SPY) and Kaggle (EUR/USD). These represent a wide range of market behaviors:
- **BTC/USD**: High volatility (crypto market)
- **EUR/USD**: High liquidity (forex market)
- **SPY ETF**: Broad U.S. equities (stock market)

Data was cleaned and formatted for machine learning:
- Removed errors and filled missing values  
- Computed RSI, MACD, OBV with `pandas-ta`  
- Normalized values between 0 and 1  
- Structured data into time-windowed sequences for LSTM training  

## 📈 Data Visualization

Our graphs were generated natively in Python using an imported library called Matplotlib. You can learn more here:
https://matplotlib.org/

We used Python 3.12, Jupyter Notebooks for visualization, and GitHub for version control. The project ran on limited hardware (Intel Gen 5 and Apple M1), so code was modularized for efficiency.

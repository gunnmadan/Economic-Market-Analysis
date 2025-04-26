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

## 💰Required Explanation for Implementation 
Our University professors require we explain our sourcing, implimentation, and code. You can find it here:
* https://github.com/gunnmadan/Machine-Learning/tree/main/data
* https://github.com/gunnmadan/Machine-Learning/tree/main/outputs
* https://github.com/gunnmadan/Machine-Learning/tree/main/scripts

My parnter, G. Madan wrote our scripts based on mathemetical models from [Professor Brian Pidgeon](https://sites.google.com/site/brianpidgeonmath9116/about-me)'s MATH3020 class. 
I focused on data gathering, documentation, script efficiency, implimentation, presentation, and communication, using the references provided previously in the [README](https://github.com/gunnmadan/Machine-Learning/) 
[Dr. Pidgeon](https://www.ratemyprofessors.com/professor/2379588)'s MATH 3020 class was integral to my personal understanding of statistics and machine learning systems. 

[Dr. Akbas](https://sites.google.com/view/esraakbas)'s exams and virtual presentations prepared us for the difficult programming challenges ahead.
The funding of the department was provided by an [NSF Career](https://csds.gsu.edu/2024/08/24/akbas-receives-nsf-career-award/) award.  
By using Python's Matplotlib library, we were able to locally compute large amounts of data in a relatively small amount of time. 
We want to thank the [GSU CATLAB](https://technology.gsu.edu/technology-services/technology-labs-printing/catlab/) for providing equal opportunities to students without hardware.

## 💼Implimentation
Providing a step-by-step guide to our project would require a full-length presentation longer than the 15-20 minutes we were given.
We would love to provide a demonstration of how Machine Learning can create economic growth.

You can find us for hire here:
[E. Munji](https://www.linkedin.com/in/ethanmunji)
[G. Madan](https://www.linkedin.com/in/gunn-madan)

We are both passionate about the new growth in the emerging fields of Data and Computer Science. 

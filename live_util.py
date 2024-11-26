import yfinance as yf
import pandas as pd
from tqdm import tqdm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings

TICKERS = [
    'SPY',   # S&P 500 ETF
    'XLB',   # Materials
    'XLE',   # Energy
    'XLF',   # Financials
    'XLI',   # Industrials
    'XLK',   # Technology
    'XLP',   # Consumer Staples
    'XLU',   # Utilities
    'XLV',   # Health Care
    'XLY',   # Consumer Discretionary
    'XLRE',  # Real Estate
    'XLC'    # Communication Services
]

today = date.today()
today_str = today.strftime("%Y-%m-%d")

today_plus1 = today + timedelta(days=1)
today_plus1_str = today_plus1.strftime("%Y-%m-%d")


def get_X_y(tickers_ls, start_date = '2017-01-01', end_date = today_str, rolling_window = 30):
    print(f"today: {today_str}. Prediction will be for open tomorrow on: {today_plus1_str}")
    returns_df = pd.DataFrame()
    for ticker in tickers_ls:
        data = yf.download(ticker, start=start_date, end=end_date)
        data["Adjustment Multiplier"] = data["Adj Close"] / data["Close"]
        data["Adj Open"] = data["Open"] * data["Adjustment Multiplier"]
        data[f"{ticker}"] = ((data["Adj Open"] - data["Adj Close"].shift(1)) / data["Adj Close"].shift(1)).fillna(0)
        ticker_returns_df = data[[f"{ticker}"]]  
        returns_df = pd.concat([returns_df, ticker_returns_df], axis=1)
        returns_df = returns_df.dropna()
    X = returns_df[['XLB', 'XLE','XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE', 'XLC' ]].shift(1).dropna()
    y = returns_df["SPY"][1:]
    return X.tail(rolling_window),y.tail(rolling_window)

def get_prediction(X,y):
    model = ARIMA(y, exog=X, order=(5, 1, 0))
    model_fit = model.fit() 
    pred_features = X.iloc[-1]
    predictions = model_fit.predict(X=pred_features)
    pred_tplus1 = predictions.iloc[0]
    print(f"The prediction for next day returns at open is: {pred_tplus1*100:.4f}%")
    return pred_tplus1



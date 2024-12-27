import yfinance as yf
import pandas as pd
from tqdm import tqdm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from datetime import date
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
today_string = today.strftime("%Y-%m-%d")

def get_X_y(tickers_ls, end_date =today_string):
    returns_df = pd.DataFrame()
    for ticker in tickers_ls:
        data = yf.download(ticker, start='2010-01-01', end=end_date)
        data["Adjustment Multiplier"] = data["Adj Close"] / data["Close"]
        data["Adj Open"]= data["Open"][ticker] * data["Adjustment Multiplier"]
        data[f"{ticker}"] = ((data["Adj Open"] - data["Adj Close"][ticker].shift(1)) / data["Adj Close"][ticker].shift(1)).fillna(0)
        ticker_returns_df = data[[f"{ticker}"]]  
        returns_df = pd.concat([returns_df, ticker_returns_df], axis=1)
        returns_df = returns_df.dropna()
    X = returns_df[['XLB', 'XLE','XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE', 'XLC' ]].shift(1).dropna()
    y = returns_df["SPY"][1:]
    return X,y

def rolling_predictions(X, y, window_size):
    predictions = []  
    dates = y.index[window_size:]  

    # Iterate over the data using a rolling window
    for end_idx in tqdm(range(window_size, len(y)), desc=f"Iterating through len(y) - {window_size}:"):
        # Define the training window
        start_idx = end_idx - window_size
        y_train = y.iloc[start_idx:end_idx]
        X_train = X.iloc[start_idx:end_idx]

        model = ARIMA(y_train, exog=X_train, order=(5, 1, 0))
        model_fit = model.fit()
        # arima_model = pm.arima.auto_arima(y=y_train, X=X_train)   
        pred_features = X.iloc[end_idx]

        # pred_features = pd.DataFrame(pred_features)
        prediction = model_fit.predict(X=pred_features)

        predictions.append(prediction.iloc[0])

    results = pd.DataFrame({'Date': dates, 'Prediction': predictions})
    results.set_index('Date', inplace=True)
    results.to_csv(f'rolling_window_predictions_{window_size}.csv')
    print(f"results saved to 'rolling_window_predictions_{window_size}.csv' !")

def generate_backtest(window_size):
    
    backtest_df = yf.download("SPY", start='2010-01-01', end=today_string)
    backtest_df = backtest_df.reset_index() 
    backtest_df["Date"] = pd.to_datetime(backtest_df["Date"]).dt.tz_localize(None)

    newbacktest_df = pd.DataFrame( { "Date" : backtest_df["Date"],
                                     "Close" : backtest_df["Close"]["SPY"],
                                     "Open" : backtest_df["Open"]["SPY"]})
    plt.figure(figsize=(14, 8))

    predictions_df = pd.read_csv(f"rolling_window_predictions_{window_size}.csv")
    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"]).dt.tz_localize(None)
    predictions_df["Prediction"] = predictions_df["Prediction"]/100
    merged_df = pd.merge(newbacktest_df, predictions_df, on="Date")
    merged_df["trade_signal"] = np.where(merged_df["Prediction"] > 0, 1, (np.where(merged_df["Prediction"] < 0, -1, 0)))
    merged_df["PnL_no_fees"] = merged_df["trade_signal"] * (merged_df["Close"].shift(1) - merged_df["Open"])
    merged_df["cum_PnL_no_fees"] = merged_df["PnL_no_fees"].cumsum()
    plt.plot(merged_df["Date"], merged_df["cum_PnL_no_fees"], label=f'Rolling Window: {window_size}')
        
    # Add legend and labels
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (no fees)")
    plt.title("Cumulative PnL for Different Rolling Window Sizes")

    output_file = f"cumulative_pnl_window_{window_size}.png"
    plt.savefig(output_file)  # Save the plot to a file
    print(f"Chart saved as {output_file} !")

def get_combined_graph(ls_of_rolling_windows):
    backtest_df = yf.download("SPY", start='2010-01-01', end=today_string)
    backtest_df = backtest_df.reset_index() 
    backtest_df["Date"] = pd.to_datetime(backtest_df["Date"]).dt.tz_localize(None)

    newbacktest_df = pd.DataFrame( { "Date" : backtest_df["Date"],
                                     "Close" : backtest_df["Close"]["SPY"],
                                     "Open" : backtest_df["Open"]["SPY"]})
    
    for roll_window in ls_of_rolling_windows:
        pred_df = pd.read_csv(f"rolling_window_predictions_{roll_window}.csv")
        pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.tz_localize(None)

        merged_df = pd.merge(newbacktest_df, pred_df, on="Date")
        merged_df["trade_signal"] = np.where(merged_df["Prediction"] > 0, 1, (np.where(merged_df["Prediction"] < 0, -1, 0)))
        merged_df["PnL_no_fees"] = merged_df["trade_signal"] * (merged_df["Close"].shift(1) - merged_df["Open"])
        merged_df["cum_PnL_no_fees"] = merged_df["PnL_no_fees"].cumsum()
        plt.plot(merged_df["Date"], merged_df["cum_PnL_no_fees"], label=f'Rolling Window {roll_window}')
    
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (no fees)")
    plt.title("Cumulative PnL for Different Rolling Window Sizes")
    output_file = f"Combined_cumulative_pnl_27122024.png"
    plt.savefig(output_file)  # Save the plot to a file
    print(f"Chart saved as {output_file} !")
    plt.show()


    


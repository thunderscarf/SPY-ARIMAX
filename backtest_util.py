import yfinance as yf
import pandas as pd
from tqdm import tqdm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import requests 
import pandas_market_calendars as mcal
import pmdarima as pm
API_KEY = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress UserWarnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings



today = date.today()
today_string = today.strftime("%Y-%m-%d")

def get_X_y(tickers_ls, end_date =today_string):
    returns_df = pd.DataFrame()
    for ticker in tickers_ls:
        data = yf.download(ticker, start="2021-01-01", end=end_date, auto_adjust=False)
        data.columns = data.columns.droplevel('Ticker')
        data["Adjustment Multiplier"] = data["Adj Close"] / data["Close"]
        data[f"{ticker}_intraday_return"] =  (data["Close"] - data["Open"]) / data["Open"] #(today's close - today's open) / today's open
        data[f"{ticker}_overnight_return"] = (data["Open"].shift(-1) - data["Close"]) / data["Close"] #(tomorrow's open - today's close)/today's close
        # data["Adj Open"]= data["Open"] * data["Adjustment Multiplier"]
        ticker_returns_df = data[[f"{ticker}_intraday_return", f"{ticker}_overnight_return"]]
        returns_df = pd.concat([returns_df, ticker_returns_df], axis=1)
        returns_df = returns_df.dropna()

    print(returns_df)
    # Use today's return for the features (X)
    X = returns_df.filter(like="_intraday_return", axis = 'columns')
    # Use tomorrow's return for the target (y)
    y = returns_df.filter(like="_overnight_return", axis = 'columns')[["SPY_overnight_return"]]
        
    # X = returns_df[['XLB', 'XLE','XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE', 'XLC' ]].shift(1).dropna()
    # y = returns_df["SPY"][1:]
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

        # model = ARIMA(y_train, exog=X_train, order=(5, 1, 0))
        # model_fit = model.fit()
        pred_features = X.iloc[[end_idx]]
        print(X_train)
        
        arima_model = pm.arima.auto_arima(y=y_train, X=X_train)   
        print(pred_features)
        y_forecast = arima_model.predict(n_periods=1, X=pred_features)
        print(y_forecast)
        # pred_features = pd.DataFrame(pred_features)
        # prediction = model_fit.predict(X=pred_features)

        predictions.append(y_forecast.iloc[0])
        print(f'Date: {y.index[end_idx]}, Prediction: {y_forecast.iloc[0]}')
    results = pd.DataFrame({'Date': dates, 'Prediction': predictions})
    results.set_index('Date', inplace=True)
    results.to_csv(f'rolling_window_predictions_{window_size}.csv')
    print(f"results saved to 'rolling_window_predictions_{window_size}.csv' !")
    return results

def generate_backtest(window_size):
    
    backtest_df = yf.download("SPY", start='2010-01-01', end=today_string, auto_adjust=False)
    backtest_df = backtest_df.reset_index() 
    backtest_df["Date"] = pd.to_datetime(backtest_df["Date"]).dt.tz_localize(None)

    newbacktest_df = pd.DataFrame( { "Date" : backtest_df["Date"],
                                     "Close" : backtest_df["Close"]["SPY"],
                                     "Open" : backtest_df["Open"]["SPY"]})
    plt.figure(figsize=(14, 8))
    print(backtest_df)
    print(newbacktest_df)
    predictions_df = pd.read_csv(f"rolling_window_predictions_{window_size}.csv")
    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"]).dt.tz_localize(None)

    merged_df = pd.merge(newbacktest_df, predictions_df, on="Date")
    merged_df["trade_signal"] = np.where(merged_df["Prediction"] > 0, 1, (np.where(merged_df["Prediction"] < 0, -1, 0)))
    merged_df["PnL_no_fees"] = merged_df["trade_signal"] * ( merged_df["Open"] - merged_df["Close"].shift(1) )
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
    return merged_df 

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
        merged_df['Prediction'] = pd.to_numeric(merged_df['Prediction'])
        merged_df["trade_signal"] = np.where(merged_df["Prediction"] > 0, 1, (np.where(merged_df["Prediction"] < 0, -1, 0)))
        merged_df["PnL_no_fees"] = merged_df["trade_signal"] * (merged_df["Open"] - merged_df["Close"].shift(1))
        merged_df["cum_PnL_no_fees"] = merged_df["PnL_no_fees"].cumsum()
        plt.plot(merged_df["Date"], merged_df["cum_PnL_no_fees"], label=f'Rolling Window {roll_window}')
    
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (no fees)")
    plt.title("Cumulative PnL for Different Rolling Window Sizes")
    output_file = f"Combined_cumulative_pnl.png"
    plt.savefig(output_file)  # Save the plot to a file
    print(f"Chart saved as {output_file} !")
    plt.show()


    

def get_options_df(ticker, date, primary_strike, fallback_strike, option_type = 'call'):
    ''' 
    option_type either 'call' or 'put'
    '''
    if option_type == None:
        return None
    try:
        url_primary = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type={option_type}&as_of={date}&strike_price={primary_strike}&limit=1000&apiKey={API_KEY}"
        response_primary = requests.get(url_primary)
        # Check if primary request is successful
        if response_primary.status_code == 200:
            data_primary = response_primary.json()
            if "results" in data_primary and data_primary["results"]:
                print(f"✅ Found results for {ticker} on {date} at strike {primary_strike}")
                return pd.json_normalize(data_primary["results"])  
        
        print(f"⚠️ No results for {ticker} at {primary_strike}, retrying with rounded strike {fallback_strike}...")
        url_fallback = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type={option_type}&as_of={date}&strike_price={fallback_strike}&limit=1000&apiKey={API_KEY}"
        response_fallback = requests.get(url_fallback)

        if response_fallback.status_code == 200:
            data_fallback = response_fallback.json()
            if "results" in data_fallback and data_fallback["results"]:
                print(f"✅ Found results for {ticker} on {date} at fallback strike {fallback_strike}")
                return pd.json_normalize(data_fallback["results"]) 

        print(f"❌ No results found for {ticker} on {date} at both strikes ({primary_strike} and {fallback_strike})")
        return None

    except Exception as e:
        print(f"Error fetching options data: {e}")
        return None
    
def get_open_close_price_option(option_ticker, entry_date):
    #get next trading day
    nyse = mcal.get_calendar('NYSE')
    next_open_day = nyse.valid_days(start_date=entry_date, end_date=entry_date + pd.Timedelta(days=10))[1]
    
    entry_date_str = entry_date.strftime('%Y-%m-%d')
    next_open_day_str = next_open_day.strftime('%Y-%m-%d')

    print(f'decision date: {entry_date_str}. trade executed at close of {entry_date_str}, end at the open of {next_open_day_str}.')
    url_primary = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{entry_date_str}/{next_open_day_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    ohlcvdf = pd.json_normalize(requests.get(url_primary).json()['results']).set_index("t")
    ohlcvdf.index = pd.to_datetime(ohlcvdf.index, unit="ms", utc=True).tz_convert("America/New_York")
    print(ohlcvdf)
    entry_price = ohlcvdf['c'].iloc[0]
    exit_price = ohlcvdf['o'].iloc[-1]
    return entry_price, exit_price, next_open_day_str


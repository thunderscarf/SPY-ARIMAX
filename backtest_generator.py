from backtest_util import *
import math 
import pandas_market_calendars as mcal
#-----------------------------------------------------------------------
window_sizes_ls = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nyse = mcal.get_calendar("NYSE")
for num in window_sizes_ls:
    # X,y = get_X_y(TICKERS)
    # print(X)
    # print(y)
    # results_df = rolling_predictions(X,y, window_size = num)
    # print(results_df)
    
    # print(backtest_df)
    # backtest_df.to_csv(f'backtest_df_{num}.csv', index=False)

    backtest_df = pd.read_csv(f'backtest_df_{num}.csv')
    backtest_df['Date'] = pd.to_datetime(backtest_df['Date'])
    backtest_df['Close_tminus1'] = backtest_df['Close'].shift(1)
    backtest_df = backtest_df.iloc[1:]

    for idx in range(len(backtest_df)):
        exit_date = backtest_df['Date'].iloc[idx]
        trade_signal = backtest_df['trade_signal'].iloc[idx]
        option_direction = 'call' if trade_signal == 1 else 'put'
        prev_close_price = backtest_df['Close_tminus1'].iloc[idx]

        date_pool = nyse.valid_days(start_date=exit_date +pd.Timedelta(days=-10), end_date=exit_date)
        entry_date = date_pool[-2]
        strike = math.floor(prev_close_price)

        options_df = get_options_df("SPY", entry_date, strike, strike+1, option_type = option_direction)

        print(f'idx: {idx}, date: {entry_date}, signal: {trade_signal}, close_price on entry_date: {prev_close_price}')
        print(options_df)

    
    # backtest_df = generate_backtest(window_size = num)


#run this after u have run the top part
# get_combined_graph([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
#-----------------------------------------------------------------------


#backtest based on voting of 3 models
# window_sizes_ls = [40]

# for num in window_sizes_ls:
#     pred_df = pd.read_csv(f"rolling_window_predictions_{num}.csv")
#     pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.tz_localize(None)
#     # pred_df[f"prediction_{num}"] = pred_df["Prediction"]
#     pred_df.rename(columns={"Prediction" : f"prediction_{num}"}, inplace=True)
#     if num == 10:
#         combined_pred_df = pred_df.copy()
#     else:
#         combined_pred_df = combined_pred_df.merge(right=pred_df, on="Date", how="inner")

# combined_pred_df["long_signal"] = np.where((combined_pred_df["prediction_10"] > 0) & (combined_pred_df["prediction_25"] > 0) & (combined_pred_df["prediction_40"] > 0), 1, 0)
# combined_pred_df["short_signal"] = np.where((combined_pred_df["prediction_10"] < 0) & (combined_pred_df["prediction_25"] < 0) & (combined_pred_df["prediction_40"] < 0), -1, 0)
# combined_pred_df["combined_signal"] = np.where(combined_pred_df["long_signal"] == 1, 1, np.where(combined_pred_df["short_signal"] == -1, -1, 0))


# backtest_df = yf.download("SPY", start='2010-01-01', end=today_string)
# backtest_df = backtest_df.reset_index() 
# backtest_df["Date"] = pd.to_datetime(backtest_df["Date"]).dt.tz_localize(None)

# newbacktest_df = pd.DataFrame( { "Date" : backtest_df["Date"],
#                                     "Close" : backtest_df["Close"]["SPY"],
#                                     "Open" : backtest_df["Open"]["SPY"]})

# merged_df = pd.merge(newbacktest_df, combined_pred_df, on="Date")
# merged_df["PnL_no_fees"] = merged_df["combined_signal"] * (merged_df["Close"].shift(1) - merged_df["Open"])
# merged_df["cum_PnL_no_fees"] = merged_df["PnL_no_fees"].cumsum()
# plt.plot(merged_df["Date"], merged_df["cum_PnL_no_fees"], label=f'Combined')
# plt.legend()
# plt.xlabel("Date")
# plt.ylabel("Cumulative PnL (no fees)")
# plt.title("Cumulative PnL for Different Rolling Window Sizes")
# plt.show()
# # print(combined_pred_df[combined_pred_df["combined_signal"] != 0])



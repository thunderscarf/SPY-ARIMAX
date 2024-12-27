from backtest_util import *
#-----------------------------------------------------------------------
# window_sizes_ls = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# for num in window_sizes_ls:
#     X,y = get_X_y(TICKERS)
#     rolling_predictions(X,y, window_size = num)
#     generate_backtest(window_size = num)


#run this after u have run the top part
# get_combined_graph([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
#-----------------------------------------------------------------------


#backtest based on voting of 3 models
window_sizes_ls = [10, 25, 40]

for num in window_sizes_ls:
    pred_df = pd.read_csv(f"rolling_window_predictions_{num}.csv")
    pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.tz_localize(None)
    # pred_df[f"prediction_{num}"] = pred_df["Prediction"]
    pred_df.rename(columns={"Prediction" : f"prediction_{num}"}, inplace=True)
    if num == 10:
        combined_pred_df = pred_df.copy()
    else:
        combined_pred_df = combined_pred_df.merge(right=pred_df, on="Date", how="inner")

combined_pred_df["long_signal"] = np.where((combined_pred_df["prediction_10"] > 0) & (combined_pred_df["prediction_25"] > 0) & (combined_pred_df["prediction_40"] > 0), 1, 0)
combined_pred_df["short_signal"] = np.where((combined_pred_df["prediction_10"] < 0) & (combined_pred_df["prediction_25"] < 0) & (combined_pred_df["prediction_40"] < 0), -1, 0)
combined_pred_df["combined_signal"] = np.where(combined_pred_df["long_signal"] == 1, 1, np.where(combined_pred_df["short_signal"] == -1, -1, 0))


backtest_df = yf.download("SPY", start='2010-01-01', end=today_string)
backtest_df = backtest_df.reset_index() 
backtest_df["Date"] = pd.to_datetime(backtest_df["Date"]).dt.tz_localize(None)

newbacktest_df = pd.DataFrame( { "Date" : backtest_df["Date"],
                                    "Close" : backtest_df["Close"]["SPY"],
                                    "Open" : backtest_df["Open"]["SPY"]})

merged_df = pd.merge(newbacktest_df, combined_pred_df, on="Date")
merged_df["PnL_no_fees"] = merged_df["combined_signal"] * (merged_df["Close"].shift(1) - merged_df["Open"])
merged_df["cum_PnL_no_fees"] = merged_df["PnL_no_fees"].cumsum()
plt.plot(merged_df["Date"], merged_df["cum_PnL_no_fees"], label=f'Combined')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Cumulative PnL (no fees)")
plt.title("Cumulative PnL for Different Rolling Window Sizes")
plt.show()
# print(combined_pred_df[combined_pred_df["combined_signal"] != 0])



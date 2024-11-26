from live_util import *


X, y = get_X_y(TICKERS, rolling_window = 40)
pred_tplus1 = get_prediction(X,y)

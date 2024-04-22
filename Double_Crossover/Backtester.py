import pandas as pd
import yfinance as yf
import talib as tl
import pytz
from datetime import datetime, date, timedelta
import numpy as np
import time

start_time = time.time()
today = date.today()
previous_day = today - timedelta(days=2)

tickers = pd.read_csv("Double_Crossover\\UsableTickers.csv", index_col=False)
tickers = tickers["tickers"].values.tolist()

data = yf.download(tickers=tickers, period="729d", interval="1h")
print(data)
cols_to_remove = ['Adj Close', 'Open', 'High', 'Low']
data = data[data.columns[~data.columns.get_level_values(0).isin(cols_to_remove)]]

tz = pytz.timezone('America/New_York')
end = datetime.combine(previous_day, datetime.min.time())
end = tz.localize(end.replace(hour=15, minute=30, second=0))
cols_to_drop = [col for col in data.columns if len(data[col].dropna()) <= 0]
data.drop(columns=cols_to_drop, inplace=True)
data = data[: end]

tickers = list(set([col[1] for col in data]))
ml_data = pd.DataFrame(columns=["Ticker", "Time", "close", "ema50_prev", "ema50", "ema200_prev", "ema200", "obv", "rsi", "outcome"])


def get_emas(dataC, dataV):
    ema50 = tl.EMA(dataC, timeperiod=50)
    ema200 = tl.EMA(dataC, timeperiod=200)
    obv = tl.OBV(dataC, dataV)
    rsi = tl.RSI(dataC)

    return ema50, ema200, obv, rsi

for ticker in tickers:
    ema50, ema200, obv, rsi = get_emas(data["Close"][ticker], data["Volume"][ticker])
    ema50_prev, ema200_prev = ema50.shift(1), ema200.shift(1)

    above = (ema50 > ema200)
    mask = above.shift(1)

    buy_signal = (above) & ~(mask.fillna(False))
    sell_signal = (ema50 < ema200)

    for i in range(len(buy_signal)):
        if(buy_signal[i]):
            in_price = data["Close"][ticker][buy_signal.index[i]]
            in_time = buy_signal.index[i]
            for j in range(i, len(buy_signal)):
                if(sell_signal[j]):
                    out_price = data["Close"][ticker][buy_signal.index[j]]
                    break
                else:
                    out_price = data["Close"][ticker][buy_signal.index[j]]
                    if(((out_price - in_price)/in_price)*100 >= 5):
                        break

            if(not np.isnan(ema200_prev[buy_signal.index[i]])):
                outcome = int(((out_price - in_price)/in_price)*100 >= 5)
                addition = {"Ticker": tickers.index(ticker), "Time": int(str(f'{in_time.hour}{in_time.minute}{in_time.second}')), "close": data["Close"][ticker][buy_signal.index[i]], "ema50_prev": ema50_prev[buy_signal.index[i]], "ema50": ema50[buy_signal.index[i]], "ema200_prev": ema200_prev[buy_signal.index[i]], "ema200": ema200[buy_signal.index[i]], "obv": obv[buy_signal.index[i]], "rsi": rsi[buy_signal.index[i]], "outcome": outcome}
                ml_data.loc[len(ml_data)] = addition

ml_data.to_csv(f"Double_Crossover\\ml_data({today}).csv", index=False)
print(f"time taken: {time.time() - start_time}")
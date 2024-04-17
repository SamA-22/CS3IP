import talib as tl
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
import pytz
import time

start_time = time.time()

def get_MACD(data):

    slow_MA = tl.EMA(data, timeperiod = 26)
    fast_MA = tl.EMA(data, timeperiod = 12)

    macd_line = fast_MA - slow_MA

    signal_line = tl.EMA(macd_line, timeperiod = 9)

    macd_dict = {
        "macd_line": macd_line,
        "signal_line": signal_line
    }

    return pd.DataFrame(macd_dict).dropna()

def get_bbands(src, length):
    upper, middle, lower = tl.BBANDS(src, length, 2.0, 2.0)

    return upper, middle, lower

def get_rsi(src):
    rsi = tl.RSI(src)

    return rsi

def calc_profitable(in_price, out_price):
    if(((out_price - in_price)/in_price)*100 >= 5):
        return 1
    else:
        return 0
    
def get_ml_data(trade_data, rsi, upperbband, lowerbband, ticker, macd):
    in_time = trade_data.index[0]
    in_price, out_price = trade_data["Close"][ticker][0], max(trade_data["High"][ticker])
    outcome = calc_profitable(in_price, out_price)

    addition = {"ticker": tickers.index(ticker), 
                "time": int(f'{in_time.hour}{in_time.minute}{in_time.second}'), 
                "close": in_price, 
                "high": data["High"][ticker][in_time], 
                "low": data["Low"][ticker][in_time], 
                "macd": macd["macd_line"][in_time], 
                "signal_line": macd["signal_line"][in_time], 
                "macd_prev": macd["macd_line_prev"][in_time], 
                "signal_line_prev": macd["signal_line_prev"][in_time],
                "rsi": rsi[in_time],
                'bband_high': upperbband[in_time], 
                'bband_low': lowerbband[in_time], 
                'outcome': outcome}
    
    return addition

def evaluate_ticker(bounds, data, macd, ticker, ml_data):
    rsi = get_rsi(data["Close"][ticker])
    upperbband, midbband, lowerbband = get_bbands(data["Close"][ticker], 20)

    maximum, minimum = bounds[0], bounds[1]

    macd["macd_line_prev"] = macd["macd_line"].shift(1)
    macd["signal_line_prev"] = macd["signal_line"].shift(1)

    #Buy signal
    buy = (macd["macd_line_prev"] < macd["signal_line_prev"]) & (macd["macd_line"] > macd["signal_line"]) & (minimum > macd["macd_line"]) & (minimum > macd["signal_line"])
    buy_signals = buy[buy]

    exit_buy = (macd["macd_line_prev"] > macd["signal_line_prev"]) & (macd["macd_line"] < macd["signal_line"]) & (0 < macd[ "macd_line"]) & (0 < macd["signal_line"])
    exit_buy = exit_buy[exit_buy]

    for i in buy_signals.index:
        if(len(exit_buy) == 0):
            trade_data = data.loc[i:data.index[-1]]
            ml_data.loc[len(ml_data)] = get_ml_data(trade_data, rsi, upperbband, lowerbband, ticker, macd)

        elif(i > exit_buy.index[-1]):
            trade_data = data.loc[i:data.index[-1]]
            ml_data.loc[len(ml_data)] = get_ml_data(trade_data, rsi, upperbband, lowerbband, ticker, macd)

        else:
            for j in exit_buy.index:
                if(j > i):
                    trade_data = data.loc[i:j]
                    ml_data.loc[len(ml_data)] = get_ml_data(trade_data, rsi, upperbband, lowerbband, ticker, macd)
                    break

tickers_data = pd.read_csv("MACD\\Ticker_Bounds.csv")
tickers = tickers_data["ticker"].values.tolist()

index = tickers_data.ticker
tickers_data = tickers_data.drop(columns="ticker")
tickers_data = tickers_data.set_index(index)


today = date.today()
previous_day = today - timedelta(days=1)

data = yf.download(tickers=tickers, period="729d", interval="1h")
cols_to_remove = ['Adj Close', 'Open']
data = data[data.columns[~data.columns.get_level_values(0).isin(cols_to_remove)]]

tz = pytz.timezone('America/New_York')
end = datetime.combine(previous_day, datetime.min.time())
end = tz.localize(end.replace(hour=15, minute=30, second=0))
cols_to_drop = [col for col in data.columns if len(data[col].dropna()) <= 0]
data.drop(columns=cols_to_drop, inplace=True)
data = data[: end]

tickers = list(set([col[1] for col in data]))
ml_data = pd.DataFrame(columns=['ticker', 'time', 'close', 'high', 'low', 'macd', 'signal_line', 'macd_prev', 'signal_line_prev', 'rsi', 'bband_high', 'bband_low', 'outcome'])

for ticker in tickers:
    macd = get_MACD(data["Close"][ticker])
    evaluate_ticker([tickers_data["upper bound"][ticker], tickers_data["lower bound"][ticker]], data, macd, ticker, ml_data)

ml_data.to_csv(f"MACD\\ml_data{today}.csv", index=False)
print(f"time taken: {time.time() - start_time}")
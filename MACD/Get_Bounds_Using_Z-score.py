import pandas as pd
import numpy as np
import talib as tl
import yfinance as yf
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
import pytz
from datetime import datetime, date, timedelta

start_time = time.time()
file = "MACD\\UsableTickers.csv"
#tickers = pd.read_csv(file).tickers.values.tolist()
tickers = ["AMD", "TSLA"]

today = date.today()
previous_day = today - timedelta(days=1)

all_ticker_bounds = pd.DataFrame(columns=["ticker", "upper bound", "lower bound"])

data = yf.download(tickers=tickers, period="729d", interval="1h")
cols_to_remove = ['Adj Close', 'Open', 'High', 'Low', 'Volume']
data = data[data.columns[~data.columns.get_level_values(0).isin(cols_to_remove)]]

tz = pytz.timezone('America/New_York')
end = datetime.combine(previous_day, datetime.min.time())
end = tz.localize(end.replace(hour=15, minute=30, second=0))
cols_to_drop = [col for col in data.columns if len(data[col].dropna()) <= 0]
data.drop(columns=cols_to_drop, inplace=True)
data = data[: end]

def get_MACD(data, ticker):
    close_data = data["Close"][ticker]

    slow_MA = tl.EMA(close_data, timeperiod = 26)
    fast_MA = tl.EMA(close_data, timeperiod = 12)

    macd_line = fast_MA - slow_MA

    signal_line = tl.EMA(macd_line, timeperiod = 9)

    macd_dict = {
        "macd_line": macd_line,
        "signal_line": signal_line
    }

    return pd.DataFrame(macd_dict).dropna()

def get_lower_outliers(sb): 
    trough_indices, _ = find_peaks(-sb)
    trough_indices = trough_indices[sb[trough_indices] < 0]
    trough_values = sb[trough_indices]

    # Calculate the Modified Z-scores
    median_val = np.median(trough_values)
    mad = np.median(np.abs(trough_values - median_val))
    modified_z_scores = 0.6745 * (trough_values - median_val) / mad

    # Set a threshold for outlier detection. Typically, a threshold of 3.5 is used.
    threshold = 3.5
    outliers = trough_values[np.abs(modified_z_scores) > threshold]
    while(outliers.shape[0] == 0):
        threshold -= 0.1
        outliers = trough_values[np.abs(modified_z_scores) > threshold]

    return trough_indices, trough_values, modified_z_scores, threshold, outliers

def get_upper_outliers(sb):
    peak_indices, _ = find_peaks(sb)
    peak_indices = peak_indices[sb[peak_indices] > 0]
    peak_values = sb[peak_indices]

    # Calculate the Modified Z-scores
    median_val = np.median(peak_values)
    mad = np.median(np.abs(peak_values - median_val))
    modified_z_scores = 0.6745 * (peak_values - median_val) / mad

    # Set a threshold for outlier detection. Typically, a threshold of 3.5 is used.
    threshold = 3.5
    outliers = peak_values[np.abs(modified_z_scores) > threshold]
    while(outliers.shape[0] == 0):
        threshold -= 0.1
        outliers = peak_values[np.abs(modified_z_scores) > threshold]

    return peak_indices, peak_values, modified_z_scores, threshold, outliers

def display_graph(mcad, upper_modified_z_scores, lower_modified_z_scores, ticker):
    plt.figure(figsize=(12, 6))
    plt.xlabel("Index")
    plt.ylabel("MACD Value")
    plt.title(ticker)
    plt.plot(mcad['macd_line'], label='macd_line', color='blue')
    plt.plot(mcad['signal_line'], label='signal_line', color='orange')
    # plt.plot(np.where(np.abs(upper_modified_z_scores) > upper_threshold)[0], upper_outliers, 'yo', markersize=6, label='Upper Outliers')
    # plt.plot(np.where(np.abs(lower_modified_z_scores) > lower_threshold)[0], lower_outliers, 'yo', markersize=6, label='Lower Outliers')
    plt.axhline(y=min(upper_outliers), color='black', linestyle='-')
    plt.axhline(y=max(lower_outliers), color='black', linestyle='-')
    plt.scatter(mcad.index, mcad['peaks'], color='green', label='Peaks', s=50, zorder=5)
    plt.scatter(mcad.index, mcad['troughs'], color='red', label='Troughs', s=50, zorder=5)
    plt.legend()
    plt.show()

for ticker in tickers:
    calc_time = time.time()
    macd = get_MACD(data, ticker)
    macd = macd.reset_index()

    peak_indices, peak_values, upper_modified_z_scores, upper_threshold, upper_outliers = get_upper_outliers(macd["signal_line"])
    trough_indices, trough_values, lower_modified_z_scores, lower_threshold, lower_outliers = get_lower_outliers(macd["signal_line"])

    # Initialize the columns with NaN values
    macd['peaks'] = np.nan
    macd['troughs'] = np.nan

    # Populate the Peaks column at the appropriate indices
    macd.loc[peak_indices, 'peaks'] = peak_values

    # Populate the Troughs column at the appropriate indices
    macd.loc[trough_indices, 'troughs'] = trough_values

    display_graph(macd, upper_modified_z_scores, lower_modified_z_scores, ticker)

    new_row = {"ticker": ticker, "upper bound": upper_outliers.min(), "lower bound": lower_outliers.max()}
    all_ticker_bounds.loc[len(all_ticker_bounds)] = new_row
    print("Ticker added")
    print(f"time taken to calculate z-score: {time.time() - calc_time}")

#all_ticker_bounds.to_csv("MACD\\Ticker_Bounds.csv", index=False)
print(f"step 3: {time.time() - start_time}")
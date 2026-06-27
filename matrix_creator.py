"""
Feature engineering and target variable module for the Pure Python Quant pipeline.

Transforms raw daily close price data (from data_fetcher.py) into a fully
engineered feature matrix with technical indicators and forward-looking target
variables for rule discovery by gini_engine.py.

Feature engineering pipeline (applied in order by build_feature_matrix):
    1. Price direction (daily up/down/same sign)
    2. Normalized SMA ratios and crossover signals (20/50/100/200-day)
    3. MACD indicator (EMA-12, EMA-26, signal line)
    4. RSI (14-day Wilder-smoothed)
    5. Binary target variables and percentage change targets for 3/90/180/365-day horizons
    6. Row cleaning (drop rows with missing core indicators, keep recent rows with None targets)

Output is saved to {ticker}_ticker/matrix.json and returned as a list of dicts.
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sma(values):
    """
    Calculate the simple arithmetic mean of a list of numeric values.

    Used as the foundation for SMA ratio calculations and as the initial seed
    value for EMA series before recursive smoothing begins.

    Parameters:
        values: Non-empty list of numeric close prices or indicator values.

    Returns:
        float: Arithmetic mean of all values in the list.
    """
    return sum(values) / len(values)


def _ema_series(values, period):
    """
    Compute an Exponential Moving Average (EMA) series for a price list.

    Initialization follows standard practice: the first EMA value at index
    (period - 1) is set to the SMA of the first `period` values. Subsequent
    values use the recursive formula:
        EMA_t = (Price_t - EMA_{t-1}) * multiplier + EMA_{t-1}
    where multiplier = 2 / (period + 1).

    All indices before the first valid EMA are set to None.

    Parameters:
        values: List of numeric close prices (length n).
        period: EMA lookback period (e.g., 12 for MACD fast line, 26 for slow).

    Returns:
        list: Length-n list of EMA values; None for indices 0 through period-2.
              Returns all-None list if len(values) < period.
    """
    n = len(values)
    ema_values = [None] * n
    if n < period:
        return ema_values

    multiplier = 2 / (period + 1)
    ema = _sma(values[:period])
    ema_values[period - 1] = ema

    for i in range(period, n):
        ema = (values[i] - ema) * multiplier + ema
        ema_values[i] = ema

    return ema_values


# ---------------------------------------------------------------------------
# Add feature functions
# ---------------------------------------------------------------------------

def add_price_direction(data):
    """
    Add daily price direction indicator based on close-to-close comparison.

    Compares each day's close to the previous day's close and assigns:
        1  → close increased (bullish day)
        -1 → close decreased (bearish day)
         0 → close unchanged
        None → first row (no prior day available)

    Modifies rows in-place by adding a "Direction" key.

    Parameters:
        data: List of row dicts with "Close" field, ordered chronologically.

    Returns:
        list: The same data list with "Direction" added to each row.
    """
    for i, row in enumerate(data):
        if i == 0:
            row["Direction"] = None
            continue

        prev_close = data[i - 1]["Close"]
        curr_close = row["Close"]

        if curr_close > prev_close:
            row["Direction"] = 1
        elif curr_close < prev_close:
            row["Direction"] = -1
        else:
            row["Direction"] = 0

    return data


def add_sma_and_crossovers(data):
    """
    Calculate 20, 50, 100 and 200 day SMA, then compute normalized SMA ratios and crossover signals.
    
    Normalized SMA ratio formula: SMA_X_Ratio = (Close - SMA_X) / SMA_X
    This represents percentage distance from the Close price, eliminating look-ahead bias.
    
    Cross_X_Y: SMA_X_Ratio > SMA_Y_Ratio is 1, otherwise 0.
    Days with insufficient data are None.
    """
    for i, row in enumerate(data):
        close_price = row["Close"]
        
        # SMA_20: last 20 close (including today)
        if i < 19:
            row["SMA_20_Ratio"] = None
            sma_20 = None
        else:
            window = [data[j]["Close"] for j in range(i - 19, i + 1)]
            sma_20 = _sma(window)
            row["SMA_20_Ratio"] = (close_price - sma_20) / sma_20

        # SMA_50: last 50 close (including today)
        if i < 49:
            row["SMA_50_Ratio"] = None
            sma_50 = None
        else:
            window = [data[j]["Close"] for j in range(i - 49, i + 1)]
            sma_50 = _sma(window)
            row["SMA_50_Ratio"] = (close_price - sma_50) / sma_50

        # SMA_100: last 100 close (including today)
        if i < 99:
            row["SMA_100_Ratio"] = None
            sma_100 = None
        else:
            window = [data[j]["Close"] for j in range(i - 99, i + 1)]
            sma_100 = _sma(window)
            row["SMA_100_Ratio"] = (close_price - sma_100) / sma_100

        # SMA_200: last 200 close (including today)
        if i < 199:
            row["SMA_200_Ratio"] = None
            sma_200 = None
        else:
            window = [data[j]["Close"] for j in range(i - 199, i + 1)]
            sma_200 = _sma(window)
            row["SMA_200_Ratio"] = (close_price - sma_200) / sma_200

        # Short / medium / long term crossover signals (based on ratios)
        if row["SMA_20_Ratio"] is None or row["SMA_50_Ratio"] is None:
            row["Cross_20_50"] = None
        else:
            row["Cross_20_50"] = 1 if row["SMA_20_Ratio"] > row["SMA_50_Ratio"] else 0

        if row["SMA_50_Ratio"] is None or row["SMA_100_Ratio"] is None:
            row["Cross_50_100"] = None
        else:
            row["Cross_50_100"] = 1 if row["SMA_50_Ratio"] > row["SMA_100_Ratio"] else 0

        if row["SMA_100_Ratio"] is None or row["SMA_200_Ratio"] is None:
            row["Cross_100_200"] = None
        else:
            row["Cross_100_200"] = 1 if row["SMA_100_Ratio"] > row["SMA_200_Ratio"] else 0

    return data


def add_macd(data):
    """
    Calculate the MACD (Moving Average Convergence Divergence) indicator.

    Computes two EMA series (12-day and 26-day) from close prices, then derives:
        - EMA_12, EMA_26: Individual exponential moving averages
        - MACD_Line: EMA_12 minus EMA_26 (momentum measure)
        - MACD_Signal: 9-day EMA of the MACD_Line (signal/trigger line)

    The signal line is initialized with the SMA of the first 9 valid MACD_Line
    values (starting at index 25, where EMA_26 first becomes available), then
    smoothed recursively using the standard EMA multiplier 2/(9+1).

    Modifies rows in-place. Intermediate EMA columns are removed during clean_matrix().

    Parameters:
        data: List of row dicts with "Close" field, ordered chronologically.

    Returns:
        list: The same data list with MACD columns added to each row.
    """
    closes = [row["Close"] for row in data]
    ema_12 = _ema_series(closes, 12)
    ema_26 = _ema_series(closes, 26)

    macd_line = [None] * len(data)
    for i in range(len(data)):
        if ema_12[i] is not None and ema_26[i] is not None:
            macd_line[i] = ema_12[i] - ema_26[i]
        data[i]["EMA_12"] = ema_12[i]
        data[i]["EMA_26"] = ema_26[i]
        data[i]["MACD_Line"] = macd_line[i]

    # MACD_Signal: 9 day EMA based on MACD_Line
    signal_period = 9
    first_macd_idx = 25  # Index of the first valid EMA_26

    for i in range(len(data)):
        data[i]["MACD_Signal"] = None

    signal_start_idx = first_macd_idx + signal_period - 1
    if len(data) > signal_start_idx:
        macd_window = [macd_line[j] for j in range(first_macd_idx, signal_start_idx + 1)]
        signal = _sma(macd_window)
        data[signal_start_idx]["MACD_Signal"] = signal

        multiplier = 2 / (signal_period + 1)
        for i in range(signal_start_idx + 1, len(data)):
            signal = (macd_line[i] - signal) * multiplier + signal
            data[i]["MACD_Signal"] = signal

    return data


def add_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI) using Wilder's smoothing method.

    For each day after the first, computes daily gain and loss from close-to-close
    change. The initial average gain/loss at index `period` uses a simple average
    of the first `period` daily gains and losses. Subsequent days apply Wilder's
    smoothed moving average:
        Avg_Gain_t = (Avg_Gain_{t-1} * (period-1) + Gain_t) / period

    RSI formula: 100 - (100 / (1 + RS)), where RS = Avg_Gain / Avg_Loss.
    When Avg_Loss is zero, RSI is set to 100.0 (maximum strength).

    Intermediate columns (Gain, Loss, Avg_Gain, Avg_Loss) are removed by clean_matrix().

    Parameters:
        data: List of row dicts with "Close" field, ordered chronologically.
        period: RSI lookback period (default: 14, industry standard).

    Returns:
        list: The same data list with "RSI" added; None for insufficient history.
    """
    n = len(data)

    for i in range(n):
        data[i]["Gain"] = None
        data[i]["Loss"] = None
        data[i]["Avg_Gain"] = None
        data[i]["Avg_Loss"] = None
        data[i]["RSI"] = None

    # Daily gain and loss (first day cannot be calculated)
    for i in range(1, n):
        change = data[i]["Close"] - data[i - 1]["Close"]
        data[i]["Gain"] = change if change > 0 else 0.0
        data[i]["Loss"] = abs(change) if change < 0 else 0.0

    if n <= period:
        return data

    # First average gain/loss: simple average from 1st day to period day
    initial_gains = [data[i]["Gain"] for i in range(1, period + 1)]
    initial_losses = [data[i]["Loss"] for i in range(1, period + 1)]
    avg_gain = _sma(initial_gains)
    avg_loss = _sma(initial_losses)

    data[period]["Avg_Gain"] = avg_gain
    data[period]["Avg_Loss"] = avg_loss

    if avg_loss == 0:
        data[period]["RSI"] = 100.0
    else:
        rs = avg_gain / avg_loss
        data[period]["RSI"] = 100 - (100 / (1 + rs))

    # Subsequent days: smoothed moving average (Wilder's smoothing)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + data[i]["Gain"]) / period
        avg_loss = (avg_loss * (period - 1) + data[i]["Loss"]) / period

        data[i]["Avg_Gain"] = avg_gain
        data[i]["Avg_Loss"] = avg_loss

        if avg_loss == 0:
            data[i]["RSI"] = 100.0
        else:
            rs = avg_gain / avg_loss
            data[i]["RSI"] = 100 - (100 / (1 + rs))

    return data

def add_target_day_percentage_change(data, days_ahead=3):
    """
    Calculate percentage price change based on average of next N days' closing prices.
    
    This function computes how much the price (on average) will change over the next N days,
    expressed as a percentage relative to today's closing price. This metric is useful for
    understanding average price movement magnitude in the short term.
    
    Formula: (Average_Future_Close - Today_Close) / Today_Close
    Where Average_Future_Close = average of the next `days_ahead` closing prices
    
    Parameters:
    - data: List of dictionaries containing daily OHLC data with 'Close' field
    - days_ahead: Number of days to look ahead for averaging (default: 3)
    
    Returns:
    - data with "Target_Day_Percentage_Change" field added
    - Field value is None if there are not enough future days available
    """
    n = len(data)
    for i in range(n):
        if i + days_ahead >=n:
            data[i]["Target_Day_Percentage_Change"] = None
        else:
            total = 0
            for j in range(days_ahead):
                total += data[i+j]["Close"]
            average_price = total/days_ahead
            average_change = (average_price-data[i]["Close"])/data[i]["Close"]
            data[i]["Target_Day_Percentage_Change"] = average_change

def add_target_90day_percentage_change(data):
    """
    Calculate percentage price change based on average closing price around 90 days in the future.
    
    This function measures the expected price movement approximately 3 months ahead by averaging
    the closing prices from days 85-95 (a 10-day window centered roughly at the 90-day mark).
    This provides a medium-term outlook for price changes and helps identify long-term trends.
    
    Formula: (Average_Close_Days_85_95 - Today_Close) / Today_Close
    Where Average_Close_Days_85_95 = average of closing prices from day i+85 to i+94
    
    Parameters:
    - data: List of dictionaries containing daily OHLC data with 'Close' field
    
    Returns:
    - data with "Target_90Day_Percentage_Change" field added
    - Field value is None if there are not enough future days available (requires at least 95 days ahead)
    """
    n = len(data)
    for i in range(n):
        if i + 95 >=n:
            data[i]["Target_90Day_Percentage_Change"] = None
        else:
            total = 0
            for j in range(85,95):
                total += data[i+j]["Close"]
            average_price = total/10
            average_change = (average_price-data[i]["Close"])/data[i]["Close"]
            data[i]["Target_90Day_Percentage_Change"] = average_change

def add_target_180day_percentage_change(data):
    """
    Calculate percentage price change based on average closing price around 180 days in the future.
    
    This function measures the expected price movement approximately 6 months ahead by averaging
    the closing prices from days 170-190 (a 20-day window centered roughly at the 180-day mark).
    This provides a long-term outlook for price changes over a 6-month horizon.
    
    Formula: (Average_Close_Days_170_190 - Today_Close) / Today_Close
    Where Average_Close_Days_170_190 = average of closing prices from day i+170 to i+189
    
    Parameters:
    - data: List of dictionaries containing daily OHLC data with 'Close' field
    
    Returns:
    - data with "Target_180Day_Percentage_Change" field added
    - Field value is None if there are not enough future days available (requires at least 190 days ahead)
    """
    n = len(data)
    for i in range(n):
        if i + 190 >=n:
            data[i]["Target_180Day_Percentage_Change"] = None
        else:
            total = 0
            for j in range(170,190):
                total += data[i+j]["Close"]
            average_price = total/20
            average_change = (average_price-data[i]["Close"])/data[i]["Close"]
            data[i]["Target_180Day_Percentage_Change"] = average_change

def add_target_365day_percentage_change(data):
    """
    Calculate percentage price change based on average closing price around 365 days in the future.
    
    This function measures the expected price movement approximately 1 year ahead by averaging
    the closing prices from days 350-380 (a 30-day window centered roughly at the 365-day mark).
    This provides a long-term outlook for annual price changes and helps identify yearly trends.
    
    Formula: (Average_Close_Days_350_380 - Today_Close) / Today_Close
    Where Average_Close_Days_350_380 = average of closing prices from day i+350 to i+379
    
    Parameters:
    - data: List of dictionaries containing daily OHLC data with 'Close' field
    
    Returns:
    - data with "Target_365Day_Percentage_Change" field added
    - Field value is None if there are not enough future days available (requires at least 380 days ahead)
    """
    n = len(data)
    for i in range(n):
        if i + 380 >=n:
            data[i]["Target_365Day_Percentage_Change"] = None
        else:
            total = 0
            for j in range(350,380):
                total += data[i+j]["Close"]
            average_price = total/30
            average_change = (average_price-data[i]["Close"])/data[i]["Close"]
            data[i]["Target_365Day_Percentage_Change"] = average_change


def add_target_variable(data, days_ahead=3):
    """
    Add a binary short-term target variable indicating future price increase.

    For each row, looks ahead up to `days_ahead` trading days. If ANY future
    day's close exceeds today's close, the target is 1 (bullish); otherwise 0.
    Also records "Which_Day_Is_Higher" — the 1-indexed day (1 to days_ahead)
    on which the price first exceeded today's close.

    Rows within `days_ahead` of the dataset end receive Target=None because
    future prices are unavailable.

    Parameters:
        data: List of row dicts with "Close" field, ordered chronologically.
        days_ahead: Number of future days to scan (default: 3).

    Returns:
        list: The same data list with "Target" and "Which_Day_Is_Higher" added.
    """
    n = len(data)

    for i in range(n):
        if i + days_ahead >= n:
            data[i]["Target"] = None
        else:
            for j in range(days_ahead):
                if data[i + j]["Close"] > data[i]["Close"]:
                    data[i]["Target"] = 1
                    data[i]["Which_Day_Is_Higher"] = j + 1
                    break
            
            if data[i].get("Target") is None:
                data[i]["Target"] = 0

    return data

def add_90_day_target_variable(data):
    """
    Add a binary 90-day target variable based on medium-term price direction.

    Compares today's close to the average close over days 85–94 ahead (a 10-day
    window centered near the 90-day mark). Target_90 = 1 if the future average
    exceeds today's close, 0 otherwise. Rows within 95 days of the end are None.

    Parameters:
        data: List of row dicts with "Close" field, ordered chronologically.

    Returns:
        list: The same data list with "Target_90" added to each row.
    """

    n = len(data)

    for i in range(n):
        if i + 95 >= n:
            data[i]["Target_90"] = None
        else:
            future_close = sum(data[j]["Close"] for j in range(i + 85, i + 95)) / 10
            data[i]["Target_90"] = 1 if future_close > data[i]["Close"] else 0

    return data

def add_180_day_target_variable(data):
    """
    Add a binary 180-day target variable based on 6-month price direction.

    Compares today's close to the average close over days 170–189 ahead (a 20-day
    window centered near the 180-day mark). Target_180 = 1 if the future average
    exceeds today's close, 0 otherwise. Rows within 190 days of the end are None.

    Parameters:
        data: List of row dicts with "Close" field, ordered chronologically.

    Returns:
        list: The same data list with "Target_180" added to each row.
    """

    n = len(data)

    for i in range(n):
        if i + 190 >= n:
            data[i]["Target_180"] = None
        else:
            future_close = sum(data[j]["Close"] for j in range(i + 170, i + 190)) / 20
            data[i]["Target_180"] = 1 if future_close > data[i]["Close"] else 0

    return data

def add_365_day_target_variable(data):
    """
    Add a binary 365-day target variable based on annual price direction.

    Compares today's close to the average close over days 350–379 ahead (a 30-day
    window centered near the 365-day mark). Target_365 = 1 if the future average
    exceeds today's close, 0 otherwise. Rows within 380 days of the end are None.

    Parameters:
        data: List of row dicts with "Close" field, ordered chronologically.

    Returns:
        list: The same data list with "Target_365" added to each row.
    """

    n = len(data)

    for i in range(n):
        if i + 380 >= n:
            data[i]["Target_365"] = None
        else:
            future_close = sum(data[j]["Close"] for j in range(i + 350, i + 380)) / 30
            data[i]["Target_365"] = 1 if future_close > data[i]["Close"] else 0

    return data

def add_all_target_variables(data):
    """
    Apply all binary target and percentage-change target functions to the data.

    Sequentially adds targets for all four horizons:
        - Binary: Target (3-day), Target_90, Target_180, Target_365
        - Percentage change: Target_Day_Percentage_Change, Target_90Day_Percentage_Change,
          Target_180Day_Percentage_Change, Target_365Day_Percentage_Change

    Binary targets are used by gini_engine for win-rate calculation; percentage
    change columns are averaged across matching rows to estimate expected profit.

    Parameters:
        data: List of row dicts with "Close" field, ordered chronologically.

    Returns:
        list: The same data list with all eight target columns added.
    """
    add_target_variable(data, days_ahead=3)
    add_90_day_target_variable(data)
    add_180_day_target_variable(data)
    add_365_day_target_variable(data)

    add_target_day_percentage_change(data, days_ahead=3)
    add_target_90day_percentage_change(data)
    add_target_180day_percentage_change(data)
    add_target_365day_percentage_change(data)

    return data


def clean_matrix(data):
    """
    Remove intermediate calculation columns and drop rows with missing core features.

    Two-phase cleanup:
        1. Strip intermediate keys (Gain, Loss, Avg_Gain, Avg_Loss, EMA_12, EMA_26)
           that were needed during RSI/MACD computation but are not used by the engine.
        2. Remove rows where any of the eleven core indicator features is None.

    Rows with None target values are deliberately KEPT so that recent market days
    (where future prices are unknown) remain available for live prediction in
    predictor.py.

    Parameters:
        data: Feature-engineered list of row dicts (modified in-place).

    Returns:
        list: Cleaned data with only valid-indicator rows remaining.
    """
    intermediate_keys = ["Gain", "Loss", "Avg_Gain", "Avg_Loss", "EMA_12", "EMA_26"]

    for row in data:
        for key in intermediate_keys:
            row.pop(key, None)

    # These are the features we absolutely need to make a prediction
    core_features = [
        "Direction", "SMA_20_Ratio", "SMA_50_Ratio", "SMA_100_Ratio", "SMA_200_Ratio", 
        "Cross_20_50", "Cross_50_100", "Cross_100_200", "MACD_Line", "MACD_Signal", "RSI"
    ]

    # Only drop the row if a CORE feature is None (keeps recent days where Target is None)
    data[:] = [
        row for row in data 
        if all(row.get(feat) is not None for feat in core_features)
    ]

    return data


def build_feature_matrix(raw_data, ticker):
    """
    Run the complete feature engineering pipeline and persist the result to disk.

    Orchestrates all feature and target computation steps in sequence:
        add_price_direction → add_sma_and_crossovers → add_macd → add_rsi
        → add_all_target_variables → clean_matrix

    Saves the final matrix to {ticker}_ticker/matrix.json with 2-space indentation.

    Parameters:
        raw_data: List of row dicts with at minimum "Date" and "Close" fields,
                  typically loaded from {ticker}_ticker/data.json.
        ticker: Stock symbol used to determine the output folder path.

    Returns:
        list: The cleaned feature matrix as a list of row dictionaries, also
              written to {ticker}_ticker/matrix.json.
    """
    add_price_direction(raw_data)
    add_sma_and_crossovers(raw_data)
    add_macd(raw_data)
    add_rsi(raw_data)
    add_all_target_variables(raw_data)
    clean_matrix(raw_data)
    folder_path = Path(f"{ticker}_ticker")
    folder_path.mkdir(parents=True, exist_ok=True)

    with open(f"{folder_path}/matrix.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    return raw_data


if __name__ == "__main__":
    with open("aapl_ticker/data.json", encoding="utf-8") as f:
        raw_data = json.load(f)

    matrix = build_feature_matrix(raw_data, ticker="aapl")
    print(f"Feature matrix ready: {len(matrix)} rows saved to aapl_matrix.json")

    if matrix:
        print("Sample row:", matrix[0])

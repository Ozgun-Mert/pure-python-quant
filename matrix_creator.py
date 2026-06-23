import json


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sma(values):
    """Calculate simple arithmetic mean of given list of numbers."""
    return sum(values) / len(values)


def _ema_series(values, period):
    """
    Calculate EMA series.
    The first EMA value is initialized with the SMA of the first `period` days.
    Multiplier = 2 / (period + 1)
    Days that cannot be calculated are None.
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
    Add direction information based on the previous day's close.
    Increase → 1, Decrease → -1, Same → 0. First day → None.
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
    Calculate MACD indicator.
    - EMA_12 and EMA_26: based on close prices (initialized with SMA)
    - MACD_Line: EMA_12 - EMA_26
    - MACD_Signal: 9 day EMA based on MACD_Line (initialized with SMA)
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
    Calculate Relative Strength Index (RSI).
    - The first `period` days's gains/losses are initialized with simple average
    - Then Wilder smoothed moving average is used for subsequent days
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


def add_target_variable(data, days_ahead=3):
    """
    Model target variable (y): until `days_ahead` any day's close is higher than today's close,
    then 1, otherwise 0. Last `days_ahead` day is None.
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
    Model target variable (y): 85-95 day's close is higher than today's close,
    then 1, otherwise 0. Last 95 day is None.
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
    Model target variable (y): 170-190 day's close is higher than today's close,
    then 1, otherwise 0. Last 190 day is None.
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
    Model target variable (y): 350-380 day's close is higher than today's close,
    then 1, otherwise 0. Last 380 day is None.
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
    add_target_variable(data, days_ahead=3)
    add_90_day_target_variable(data)
    add_180_day_target_variable(data)
    add_365_day_target_variable(data)

    return data


def clean_matrix(data):
    """
    Remove rows with None and clean intermediate calculation keys.
    Return only the features required for the model.
    """
    intermediate_keys = ["Gain", "Loss", "Avg_Gain", "Avg_Loss", "EMA_12", "EMA_26"]

    for row in data:
        for key in intermediate_keys:
            row.pop(key, None)

    data[:] = [row for row in data if all(value is not None for value in row.values())]

    return data


def build_feature_matrix(raw_data, file_path="final_matrix.json"):
    """
    Run all feature engineering steps sequentially,
    save the cleaned matrix to `file_path`.json and return the final matrix as a list of dictionaries.
    """
    add_price_direction(raw_data)
    add_sma_and_crossovers(raw_data)
    add_macd(raw_data)
    add_rsi(raw_data)
    add_all_target_variables(raw_data)
    clean_matrix(raw_data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    return raw_data


if __name__ == "__main__":
    with open("aapl_data.json", encoding="utf-8") as f:
        raw_data = json.load(f)

    matrix = build_feature_matrix(raw_data, file_path="final_matrix.json")
    print(f"Feature matrix ready: {len(matrix)} rows saved to final_matrix.json")

    if matrix:
        print("Sample row:", matrix[0])

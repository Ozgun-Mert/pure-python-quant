"""
Market data acquisition module for the Pure Python Quant pipeline.

This module handles downloading historical stock price data from Yahoo Finance
via the yfinance library and converting the raw pandas DataFrame into a clean
list of dictionaries suitable for downstream feature engineering in
matrix_creator.py.

Each record in the output contains:
    - Date: Trading date as a YYYY-MM-DD string
    - Close: Adjusted closing price as a float

The fetch_data() function is the primary entry point; it orchestrates download,
parsing, and type normalization in a single call.
"""

import json
from datetime import datetime
import yfinance as yf
from pathlib import Path


def download_data(ticker, start_date, end_date):
    """
    Download historical OHLCV market data from Yahoo Finance for a given ticker.

    Uses the yfinance library to retrieve daily price bars between the specified
    start and end dates (inclusive of start, exclusive of end per yfinance convention).
    Progress output is suppressed for cleaner console logs during batch runs.

    Parameters:
        ticker: Stock symbol to download (e.g., "AAPL", "MSFT"). Case-insensitive
                for most Yahoo Finance symbols.
        start_date: Start date as a string in YYYY-MM-DD format.
        end_date: End date as a string in YYYY-MM-DD format.

    Returns:
        pandas.DataFrame: Raw download result containing Date index and OHLCV columns.
                          May be empty if the ticker is invalid or the date range
                          has no trading data.
    """
    return yf.download(ticker, start=start_date, end=end_date, progress=False)


def _to_float(value):
    """
    Normalize a close price value to a plain Python float.

    Yahoo Finance may return scalar values as pandas Series (single-row) when
    downloading a single ticker. This helper extracts the first element via
    .iloc[0] when needed, otherwise casts directly to float.

    Parameters:
        value: A numeric scalar, pandas Series, or other value representing a
               closing price.

    Returns:
        float: The closing price as a standard Python float.
    """
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _to_date_string(date):
    """
    Normalize a date index value to a YYYY-MM-DD string.

    Handles pandas Timestamp objects (via strftime) and falls back to string
    truncation for other date-like types to ensure consistent date formatting
    across the pipeline.

    Parameters:
        date: A pandas Timestamp, datetime, or string-like date value from the
              DataFrame index.

    Returns:
        str: Date formatted as "YYYY-MM-DD".
    """
    if hasattr(date, "strftime"):
        return date.strftime("%Y-%m-%d")
    return str(date)[:10]


def parse_data(data):
    """
    Convert a yfinance pandas DataFrame into a list of row dictionaries.

    Iterates over each trading day in the DataFrame and extracts only the Date
    and Close columns, which are the minimum fields required by matrix_creator.py.
    Rows from empty or None DataFrames produce an empty list.

    Parameters:
        data: pandas DataFrame returned by download_data(), indexed by date with
              at minimum a "Close" column.

    Returns:
        list[dict]: Ordered list of records, each with keys "Date" (str) and
                    "Close" (float). Returns [] if data is None or empty.
    """
    if data is None or data.empty:
        return []

    records = []
    for date, row in data.iterrows():
        records.append(
            {
                "Date": _to_date_string(date),
                "Close": _to_float(row["Close"]),
            }
        )

    return records


def fetch_data(ticker, start_date, end_date):
    """
    Download, parse, and return clean market data for a ticker and date range.

    This is the main public API of the module. It chains download_data() and
    parse_data() to produce a JSON-serializable list of daily close prices
    ready for storage in {ticker}_ticker/data.json.

    Parameters:
        ticker: Stock symbol to fetch (e.g., "aapl").
        start_date: Start date as YYYY-MM-DD string.
        end_date: End date as YYYY-MM-DD string.

    Returns:
        list[dict]: Cleaned daily records with "Date" and "Close" keys.
    """
    raw_data = download_data(ticker, start_date, end_date)
    return parse_data(raw_data)


if __name__ == "__main__":
    ticker = input("Enter a ticker: ").strip()
    start_date = "2020-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"Downloading data for {ticker}...")
    data = fetch_data(ticker, start_date, end_date)

    folder_path = Path(f"{ticker}_ticker")
    folder_path.mkdir(parents=True, exist_ok=True)

    output_file = "data.json"
    with open(f"{folder_path}/{output_file}", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} records to {output_file}")

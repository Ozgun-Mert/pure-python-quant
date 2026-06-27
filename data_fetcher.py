import json
from datetime import datetime
import yfinance as yf
from pathlib import Path



def download_data(ticker, start_date, end_date):
    """Download data from Yahoo Finance."""
    return yf.download(ticker, start=start_date, end=end_date, progress=False)


def _to_float(value):
    """Convert close value to float."""
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _to_date_string(date):
    """Convert date to YYYY-MM-DD string."""
    if hasattr(date, "strftime"):
        return date.strftime("%Y-%m-%d")
    return str(date)[:10]


def parse_data(data):
    """Convert pandas DataFrame to list of dictionaries."""
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
    """Download data, parse it and return clean list."""
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

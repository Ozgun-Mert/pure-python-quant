import json
from datetime import datetime
import yfinance as yf


def download_data(ticker, start_date, end_date):
    """Ham hisse verisini yfinance ile indirir."""
    return yf.download(ticker, start=start_date, end=end_date, progress=False)


def _to_float(value):
    """Close değerini Series veya skaler olsun, güvenli şekilde float'a çevirir."""
    if hasattr(value, "iloc"):
        return float(value.iloc[0])
    return float(value)


def _to_date_string(date):
    """Pandas Timestamp veya benzeri tarih nesnesini YYYY-MM-DD string'e çevirir."""
    if hasattr(date, "strftime"):
        return date.strftime("%Y-%m-%d")
    return str(date)[:10]


def parse_data(data):
    """Pandas DataFrame'i saf Python sözlükler listesine dönüştürür."""
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
    """Veriyi indirir, parse eder ve temiz listeyi döndürür."""
    raw_data = download_data(ticker, start_date, end_date)
    return parse_data(raw_data)


if __name__ == "__main__":
    ticker = input("Enter a ticker: ").strip()
    start_date = "2020-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"Downloading data for {ticker}...")
    data = fetch_data(ticker, start_date, end_date)

    output_file = f"{ticker.lower()}_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} records to {output_file}")

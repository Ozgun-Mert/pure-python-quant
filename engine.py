"""
Unified orchestrator for the Pure Python Quant pipeline.

Runs the full in-memory workflow: fetch market data, build feature matrix,
discover trading rules via the Gini Engine, select champion signals, and
generate predictions for a given date.

JSON files are never written by this module. Use individual module __main__
blocks for standalone debugging with disk persistence.
"""

import sys
from datetime import datetime, timedelta
from typing import List, Optional

from data_fetcher import fetch_data
from matrix_creator import build_feature_matrix
from gini_engine import gini_engine
from best_signal_finder import gini_output_to_values, finder
from predictor import generate_predictions
from classes import Prediction

DEFAULT_START_DATE = "2020-01-01"


def engine(ticker: str, target_date: Optional[str] = None) -> List[Prediction]:
    """
    Run the full quant pipeline in memory and return predictions for a date.

    Pipeline steps:
        1. Fetch historical price data from Yahoo Finance
        2. Build feature matrix with technical indicators and targets
        3. Discover trading rules via exhaustive Gini Engine analysis
        4. Select champion bullish/bearish signals per horizon
        5. Evaluate champion rules against the target date's market data

    Parameters:
        ticker: Stock symbol (e.g., "aapl").
        target_date: ISO date string (YYYY-MM-DD). Defaults to today.

    Returns:
        List[Prediction]: Predictions for champion signals triggered on target_date.

    Raises:
        ValueError: If no market data could be fetched for the ticker.
    """
    if target_date is None:
        target_date = datetime.today().strftime("%Y-%m-%d")

    end_date = (
        datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    raw_data = fetch_data(ticker, DEFAULT_START_DATE, end_date)
    if not raw_data:
        raise ValueError(f"No market data returned for ticker '{ticker}'.")

    matrix = build_feature_matrix(raw_data)

    gini_results = gini_engine(matrix)
    champions = finder(
        gini_output_to_values(gini_results["Target"]),
        gini_output_to_values(gini_results["Target_90"]),
        gini_output_to_values(gini_results["Target_180"]),
        gini_output_to_values(gini_results["Target_365"]),
    )

    return generate_predictions(ticker, target_date, matrix, champions)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python engine.py <ticker> [date]")
        print("  date defaults to today (YYYY-MM-DD)")
        sys.exit(1)

    cli_ticker = sys.argv[1]
    cli_date = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Running pipeline for {cli_ticker.upper()}...")
    predictions = engine(cli_ticker, cli_date)

    date_label = cli_date or datetime.today().strftime("%Y-%m-%d")
    print("-" * 70)
    if not predictions:
        print(f"No champion trading signals were triggered on {date_label}.")
    else:
        print(f"Found {len(predictions)} active trading signals for {date_label}:")
        for p in predictions:
            print(p)
            print(f"   Conditions met: {', '.join(p.rules_matched)}")
    print("-" * 70)

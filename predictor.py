"""
Live prediction module for the Pure Python Quant pipeline.

This module evaluates champion trading signals (selected by best_signal_finder)
against a specific day's market data from the feature matrix. When all rules
in a champion signal are satisfied, a Prediction object is generated with
the signal direction, probability, expected price change, and time horizon.

Workflow:
    1. Load the feature matrix row for the target ticker and date
    2. Retrieve all eight champion signals via best_signal_finder.finder()
    3. Evaluate each champion's Rule conditions against the market row
    4. Return Prediction objects for every champion whose rules all match

The probability assigned to each prediction is capped at 0.8 to avoid over-
confidence from historically high win rates.
"""

import json
import operator
from typing import List, Optional
from dataclasses import dataclass

from classes import Value, Type, Rule, Prediction
from best_signal_finder import finder

OPS = {
    "==": operator.eq,
    "<=": operator.le,
    ">=": operator.ge,
    "<":  operator.lt,
    ">":  operator.gt
}
"""
Mapping of comparison operator strings to Python operator callables.

Mirrors the OPS dictionary in gini_engine.py. Used to evaluate Rule conditions
against live market data rows during prediction generation.
"""

def evaluate_rule(rule: Rule, row_data: dict) -> bool:
    """
    Test whether a single Rule's condition is satisfied by a market data row.

    Looks up the rule's feature value in the row dictionary and applies the
    rule's comparison operator against the rule's threshold. Returns False
    if the feature value is missing (None) or if the operator is unrecognized.

    Parameters:
        rule: Rule object containing feature name, threshold, and operation.
        row_data: Single day's feature matrix row as a dictionary (from matrix.json).

    Returns:
        bool: True if the condition evaluates to True, False otherwise.

    Raises:
        ValueError: If rule.operation is not a key in the OPS mapping.
    """
    feature_val = row_data.get(rule.feature)
    
    if feature_val is None:
        return False
    
    op_func = OPS.get(rule.operation)
    if not op_func:
        raise ValueError(f"Unknown comparison operation: {rule.operation}")
        
    return op_func(feature_val, rule.threshold)

def get_market_data_for_date(ticker: str, target_date: str) -> Optional[dict]:
    """
    Retrieve a single day's feature matrix row for a ticker and date.

    Loads the full matrix.json file for the ticker and scans for a row whose
    "Date" field matches the target_date string exactly.

    Parameters:
        ticker: Stock symbol (e.g., "aapl") whose matrix file to read.
        target_date: ISO date string (YYYY-MM-DD) to look up.

    Returns:
        dict: The matching row dictionary with all feature and target columns,
              or None if the file is missing or no row matches the date.
    """
    file_path = f"{ticker}_ticker/matrix.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Market matrix data for {ticker.upper()} not found. Please run matrix_creator.py first.")
        return None

    for row in data:
        if row.get("Date") == target_date:
            return row
            
    return None

def generate_predictions(ticker: str, target_date: str) -> List[Prediction]:
    """
    Generate live trading predictions for a ticker on a specific date.

    For each of the eight champion signals (HIGH/LOW across four horizons),
    evaluates all constituent Rule conditions against the target date's market
    data. When every rule in a champion signal passes, creates a Prediction
    with:
        - signal_type: "HIGH" or "LOW" from the champion Value
        - probability: min(win_rate, 0.8) to cap over-confidence
        - expected_change_pct: champion's historical average profit
        - expected_days: champion's prediction horizon (3, 90, 180, or 365)
        - rules_matched: string representations of each satisfied Rule

    Parameters:
        ticker: Stock symbol to predict for (e.g., "aapl").
        target_date: ISO date string (YYYY-MM-DD) of the market day to evaluate.

    Returns:
        List[Prediction]: Zero or more Prediction objects for champion signals
                          whose rules all fired on the target date. Empty list
                          if market data is unavailable or no rules match.
    """
    row_data = get_market_data_for_date(ticker, target_date)
    if not row_data:
        print(f"Warning: No market data found for {ticker.upper()} on {target_date}.")
        return []

    try:
        champions_dict = finder(ticker)
    except Exception as e:
        print(f"Error retrieving champion rules: {e}")
        return []

    predictions = []

    for champ_key, champion_value in champions_dict.items():
        if champion_value is None:
            continue
            
        all_rules_met = True
        for rule in champion_value.rules:
            if not evaluate_rule(rule, row_data):
                all_rules_met = False
                break
                
        if all_rules_met:
            pred = Prediction(
                ticker=ticker,
                date=target_date,
                signal_type=champion_value.type.name,
                probability=min(champion_value.win_rate, 0.8),
                expected_change_pct=champion_value.percentage_profit,
                expected_days=champion_value.day,
                rules_matched=[str(r) for r in champion_value.rules]
            )
            predictions.append(pred)

    return predictions


if __name__ == "__main__":
    test_ticker = "aapl"
    
    test_date = "2026-06-03"  
    
    print(f"Scanning market conditions for {test_ticker.upper()} on {test_date}...")
    results = generate_predictions(test_ticker, test_date)
    
    print("-" * 70)
    if not results:
        print("No champion trading signals were triggered on this date.")
    else:
        print(f"Found {len(results)} active trading signals:")
        for p in results:
            print(p)
            print(f"   L_ Conditions met: {', '.join(p.rules_matched)}")
    print("-" * 70)
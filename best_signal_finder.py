"""
Champion signal selection module for the Pure Python Quant pipeline.

After the Gini Engine discovers thousands of multi-feature trading rules,
this module filters, ranks, and selects the single best ("champion") bullish
and bearish signal for each prediction horizon.

Selection criteria vary by horizon:
    - 3-day (Target): High win rate (95%+), moderate profit thresholds
    - 90-day: Higher profit expectations for bullish signals
    - 180-day: Relaxed win rate (80%) with higher support requirements
    - 365-day: Long-term profit and support thresholds

The finder() function is the main orchestrator: it accepts in-memory rule lists,
applies horizon-specific filters, and returns eight champion Value objects
(one HIGH and one LOW per horizon).
"""

import json
from typing import List
from classes import Value, Type

def filter_signals(signals: List[Value], signal_type: Type, min_win_rate: float, min_support: int, top_n: int, min_profit: float) -> List[Value]:
    """
    Filter and rank trading signals by quality metrics for a given direction.

    Applies minimum win rate and support thresholds, then sorts by profit in a
    direction-aware manner:
        - HIGH (bullish): Sort descending by percentage_profit; keep signals
          with profit >= min_profit (positive expected gain).
        - LOW (bearish): Sort ascending by percentage_profit; keep signals
          with profit <= min_profit (negative expected decline).

    Parameters:
        signals: Full list of Value objects to filter (typically all rules
                 from one target JSON file).
        signal_type: Type.HIGH for bullish or Type.LOW for bearish filtering.
        min_win_rate: Minimum historical win rate (0.0–1.0) required to pass.
        min_support: Minimum number of historical occurrences required.
        top_n: Maximum number of signals to return after sorting.
        min_profit: Profit threshold — lower bound for HIGH, upper bound for LOW.

    Returns:
        List[Value]: Up to top_n signals meeting all criteria, sorted by profit.
    """
    filtered_signals = [
        v for v in signals 
        if v.win_rate >= min_win_rate 
        and v.support >= min_support 
        and v.type == signal_type
    ]

    if signal_type == Type.HIGH:
        sorted_signals = sorted(filtered_signals, key=lambda v: v.percentage_profit, reverse=True)
        sorted_signals = [
            v for v in sorted_signals
            if v.percentage_profit>=min_profit
        ]
    else:
        sorted_signals = sorted(filtered_signals, key=lambda v: v.percentage_profit, reverse=False)
        sorted_signals = [
            v for v in sorted_signals
            if v.percentage_profit<=min_profit
        ]

    return sorted_signals[:top_n]

def find_champion(signals: List[Value], signal_type: Type) -> Value:
    """
    Select the single best signal from a pre-filtered candidate list.

    Uses a lexicographic ranking tuple to break ties deterministically:
        - HIGH champions: maximize (win_rate, support, percentage_profit, -rule_count)
          Prefer fewer rules when other metrics are equal (Occam's razor).
        - LOW champions: maximize (win_rate, support, -percentage_profit, rule_count)
          Prefer more negative profit (stronger bearish signal) and fewer rules
          when tied on earlier criteria.

    Parameters:
        signals: Pre-filtered list of candidate Value objects (from filter_signals).
        signal_type: Type.HIGH or Type.LOW — determines the ranking key logic.

    Returns:
        Value: The highest-ranked signal, or None if the input list is empty.
    """
    if not signals:
        return None
    
    if len(signals) == 1:
        return signals[0]
    
    if signal_type == Type.HIGH:
        champion = max(
            signals, key= lambda v: (
                v.win_rate,
                v.support,
                v.percentage_profit,
                -len(v.rules)
            )
        )
    else:
        champion = max(
            signals, key= lambda v: (
                v.win_rate,
                v.support,
                -v.percentage_profit,
                len(v.rules)
            )
        )

    return champion


def best_for_target(target_list):
    """
    Filter and rank signals for the 3-day (short-term) prediction horizon.

    Applies strict bullish criteria (95% win rate, 1%+ profit) and moderate
    bearish criteria (40% win rate, up to -0.5% profit). Returns the top
    candidates for champion selection.

    Parameters:
        target_list: List of Value objects loaded from Target.json.

    Returns:
        dict: Keys "high" and "low", each mapping to a filtered List[Value].
    """
    best_highs = filter_signals(
        signals=target_list, 
        signal_type=Type.HIGH, 
        min_win_rate=0.95, 
        min_support=10, 
        top_n=10,
        min_profit=0.01
    )

    #print(best_highs)

    best_lows = filter_signals(
        signals=target_list, 
        signal_type=Type.LOW, 
        min_win_rate=0.40, 
        min_support=10, 
        top_n=5,
        min_profit=-0.005
    )

    #print(best_lows)

    return {
        "high": best_highs,
        "low": best_lows
    }

def best_for_90_target(target_90_list):
    """
    Filter and rank signals for the 90-day (medium-term) prediction horizon.

    Bullish signals require 95% win rate and 30%+ average profit over 90 days.
    Bearish signals require only 20% win rate but expect at least -10% decline.

    Parameters:
        target_90_list: List of Value objects loaded from Target_90.json.

    Returns:
        dict: Keys "high" and "low", each mapping to a filtered List[Value].
    """
    best_highs = filter_signals(
        signals=target_90_list, 
        signal_type=Type.HIGH, 
        min_win_rate=0.95, 
        min_support=10, 
        top_n=10,
        min_profit=0.3
    )

    #print(best_highs)

    best_lows = filter_signals(
        signals=target_90_list, 
        signal_type=Type.LOW, 
        min_win_rate=0.20, 
        min_support=10, 
        top_n=5,
        min_profit=-0.1
    )

    #print(best_lows)

    return {
        "high": best_highs,
        "low": best_lows
    }

def best_for_180_target(target_180_list):
    """
    Filter and rank signals for the 180-day (6-month) prediction horizon.

    Bullish criteria are relaxed to 80% win rate with 20+ support and 30%+
    profit. Bearish criteria mirror the 90-day horizon.

    Parameters:
        target_180_list: List of Value objects loaded from Target_180.json.

    Returns:
        dict: Keys "high" and "low", each mapping to a filtered List[Value].
    """
    best_highs = filter_signals(
        signals=target_180_list, 
        signal_type=Type.HIGH, 
        min_win_rate=0.80, 
        min_support=20, 
        top_n=10,
        min_profit=0.30
    )
    
    #print(best_highs)
    
    best_lows = filter_signals(
        signals=target_180_list, 
        signal_type=Type.LOW, 
        min_win_rate=0.20, 
        min_support=10, 
        top_n=5,
        min_profit=-0.1
    )

    #print(best_lows)

    return {
        "high": best_highs,
        "low": best_lows
    }

def best_for_365_target(target_365_list):
    """
    Filter and rank signals for the 365-day (annual) prediction horizon.

    Bullish signals require 95% win rate, 20+ support, and 40%+ annual profit.
    Bearish signals require 40% win rate with up to -40% expected decline.

    Parameters:
        target_365_list: List of Value objects loaded from Target_365.json.

    Returns:
        dict: Keys "high" and "low", each mapping to a filtered List[Value].
    """
    best_highs = filter_signals(
        signals=target_365_list, 
        signal_type=Type.HIGH, 
        min_win_rate=0.95, 
        min_support=20, 
        top_n=10,
        min_profit=0.4
    )

    #print(best_highs)

    best_lows = filter_signals(
        signals=target_365_list, 
        signal_type=Type.LOW, 
        min_win_rate=0.40, 
        min_support=10, 
        top_n=5,
        min_profit=-0.4
    )

    #print(best_lows)

    return {
        "high": best_highs,
        "low": best_lows
    }

def gini_output_to_values(gini_output: dict) -> List[Value]:
    """
    Convert in-memory Gini Engine output to a list of Value objects.

    Parameters:
        gini_output: Dict mapping rule identifiers to serialized Value dicts,
                     as returned by gini_engine() for one target horizon.

    Returns:
        List[Value]: All discovered signals as Value instances.
    """
    return [Value.from_dict(val_dict) for val_dict in gini_output.values()]


def load_json_to_values(file_path: str) -> List[Value]:
    """
    Load a Gini Engine output JSON file and deserialize all rules into Value objects.

    The JSON files store rules as a dictionary keyed by rule identifiers
    (e.g., "HIGH-Direction-RSI"). Each value is a serialized Value dict that
    is converted via Value.from_dict().

    Parameters:
        file_path: Path to a target JSON file (e.g., "aapl_ticker/Target.json").

    Returns:
        List[Value]: All discovered signals from the file as Value instances.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    
    return [Value.from_dict(val_dict) for val_dict in target_data.values()]

def read_all_data(ticker: str):
    """
    Load all four horizon-specific rule files for a given ticker.

    Reads Target.json, Target_90.json, Target_180.json, and Target_365.json
    from the ticker's data folder and deserializes each into Value object lists.

    Parameters:
        ticker: Stock symbol whose rule files should be loaded (e.g., "aapl").

    Returns:
        tuple: Four lists of Value objects in order:
               (target_list, target_90_list, target_180_list, target_365_list).
    """
    target_list = load_json_to_values(f"{ticker}_ticker/Target.json")
    target_90_list = load_json_to_values(f"{ticker}_ticker/Target_90.json")
    target_180_list = load_json_to_values(f"{ticker}_ticker/Target_180.json")
    target_365_list = load_json_to_values(f"{ticker}_ticker/Target_365.json")

    return target_list, target_90_list, target_180_list, target_365_list

def finder(
    target_list: List[Value],
    target_90_list: List[Value],
    target_180_list: List[Value],
    target_365_list: List[Value],
):
    """
    Discover the champion bullish and bearish signals across all prediction horizons.

    Applies horizon-specific filtering via best_for_* functions, then selects a
    single champion from each filtered list via find_champion().

    Returns eight champion Value objects — one HIGH and one LOW for each of the
    four horizons (3-day, 90-day, 180-day, 365-day). Any horizon with no
    qualifying signals yields None for that champion slot.

    Parameters:
        target_list: Value objects for the 3-day horizon.
        target_90_list: Value objects for the 90-day horizon.
        target_180_list: Value objects for the 180-day horizon.
        target_365_list: Value objects for the 365-day horizon.

    Returns:
        dict: Eight keys mapping to champion Value objects (or None):
              champion_target_high/low, champion_90_target_high/low,
              champion_180_target_high/low, champion_365_target_high/low.
    """
    best_target = best_for_target(target_list=target_list)
    best_90_target = best_for_90_target(target_90_list=target_90_list)
    best_180_target = best_for_180_target(target_180_list=target_180_list)
    best_365_target = best_for_365_target(target_365_list=target_365_list)

    champion_target_high = find_champion(signals=best_target["high"], signal_type=Type.HIGH)
    champion_target_low = find_champion(signals=best_target["low"], signal_type=Type.LOW)
    champion_90_target_high = find_champion(signals=best_90_target["high"], signal_type=Type.HIGH)
    champion_90_target_low = find_champion(signals=best_90_target["low"], signal_type=Type.LOW)
    champion_180_target_high = find_champion(signals=best_180_target["high"], signal_type=Type.HIGH)
    champion_180_target_low = find_champion(signals=best_180_target["low"], signal_type=Type.LOW)
    champion_365_target_high = find_champion(signals=best_365_target["high"], signal_type=Type.HIGH)
    champion_365_target_low = find_champion(signals=best_365_target["low"], signal_type=Type.LOW)

    return {
        "champion_target_high": champion_target_high,
        "champion_target_low": champion_target_low,
        "champion_90_target_high": champion_90_target_high,
        "champion_90_target_low": champion_90_target_low,
        "champion_180_target_high": champion_180_target_high,
        "champion_180_target_low": champion_180_target_low,
        "champion_365_target_high": champion_365_target_high,
        "champion_365_target_low": champion_365_target_low
    }


def finder_from_ticker(ticker: str):
    """
    Load rule files from disk and discover champion signals for a ticker.

    Standalone wrapper that reads Target*.json files before calling finder().
    Used only when running best_signal_finder.py directly or from predictor.py __main__.

    Parameters:
        ticker: Stock symbol to analyze (e.g., "aapl").

    Returns:
        dict: Eight champion Value objects (or None) — same as finder().
    """
    target_list, target_90_list, target_180_list, target_365_list = read_all_data(ticker)
    return finder(target_list, target_90_list, target_180_list, target_365_list)

if __name__ == "__main__":
    champions = finder_from_ticker(ticker='aapl')
    for key, champion in champions.items():
        print(f"{key}: {champion}")
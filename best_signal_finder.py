import json
from typing import List
from classes import Value, Type

def filter_signals(signals: List[Value], signal_type: Type, min_win_rate: float, min_support: int, top_n: int, min_profit: float) -> List[Value]:
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

def best_for_target(target_list):
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

def load_json_to_values(file_path: str) -> List[Value]:
    with open(file_path, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    
    return [Value.from_dict(val_dict) for val_dict in target_data.values()]

def read_all_data(ticker: str):
    target_list = load_json_to_values(f"{ticker}_ticker/Target.json")
    target_90_list = load_json_to_values(f"{ticker}_ticker/Target_90.json")
    target_180_list = load_json_to_values(f"{ticker}_ticker/Target_180.json")
    target_365_list = load_json_to_values(f"{ticker}_ticker/Target_365.json")

    return target_list, target_90_list, target_180_list, target_365_list

def finder(ticker: str):
    target_list, target_90_list, target_180_list, target_365_list = read_all_data(ticker)

    best_target = best_for_target(target_list=target_list)
    best_90_target = best_for_90_target(target_90_list=target_90_list)
    best_180_target = best_for_180_target(target_180_list=target_180_list)
    best_365_target = best_for_365_target(target_365_list=target_365_list)

if __name__ == "__main__":
    finder(ticker='aapl')

        
    
import test_creator
import json
import operator
from classes import Type, Rule, Value

features_to_keep = ["Direction", "SMA_20_Ratio", "SMA_50_Ratio", "SMA_100_Ratio", "SMA_200_Ratio", "Cross_20_50", "Cross_50_100", "Cross_100_200", "MACD_Line", "MACD_Signal", "RSI"]

OPS = {
    "==": operator.eq,
    "<=": operator.le,
    ">=": operator.ge,
    "<":  operator.lt,
    ">":  operator.gt
}
"""
Mapping of operator symbols to Python operator functions.
Used to evaluate conditions like 'RSI > 70' or 'SMA_20_Ratio <= 0.1'
"""

TARGET_CONFIG = {
    "Target": {
        "days": 3,
        "pct_col": "Target_Day_Percentage_Change",
        "day_col": "Which_Day_Is_Higher"
    },
    "Target_90": {
        "days": 90,
        "pct_col": "Target_90Day_Percentage_Change",
        "day_col": None
    },
    "Target_180": {
        "days": 180,
        "pct_col": "Target_180Day_Percentage_Change",
        "day_col": None
    },
    "Target_365": {
        "days": 365,
        "pct_col": "Target_365Day_Percentage_Change",
        "day_col": None
    }
}
# Configuration for different target prediction horizons.
# Each target configuration defines:
# - days: Time horizon in days for the prediction
# - pct_col: Column name containing the average percentage change for this horizon
# - day_col: Column name containing which day the signal triggered (None if not applicable)
# These configurations map to the target variables created in matrix_creator.py,
# allowing the engine to analyze multiple prediction timeframes simultaneously.

def get_best_single_thresholds(raw_data, features_to_keep, target_col, min_samples=100):
    """
    Find optimal threshold values for each feature that maximize predictive power for a target variable.
    
    This function analyzes each feature individually to determine the best threshold values that
    separate bullish (target=1) from bearish (target=0) market conditions. For categorical features
    (5 or fewer unique values), it tests equality conditions. For continuous features, it tests
    both "<=" and ">" conditions and selects the top performing thresholds.
    
    The process:
    1. For each feature, extract all non-null values
    2. Determine if the feature is categorical (<=5 unique values) or continuous
    3. For categorical: test equality with each unique value
    4. For continuous: test all possible split points with <= and > operators
    5. Calculate bullish and bearish probabilities for each threshold/operator combination
    6. For continuous features with many candidates, select top 5 by probability and average their thresholds
    
    Parameters:
    - raw_data: List of dictionaries containing feature columns and target column
    - features_to_keep: List of feature names to analyze (strings)
    - target_col: Name of the target column (string) containing 0/1 values
    - min_samples: Minimum number of samples required for a valid threshold (default: 100)
    
    Returns:
    - Dictionary with feature names as keys and values as dicts containing:
      - "high": List of thresholds/operators for bullish conditions
      - "low": List of thresholds/operators for bearish conditions
      Each threshold dict contains "threshold" (float) and "operation" (string: "==", "<=", ">")
    """
    results = {}
    for feature in features_to_keep:
        feature_values = [row[feature] for row in raw_data if feature in row and row[feature] is not None]
        if not feature_values:
            continue
            
        unique_values = sorted(set(feature_values))
        is_categorical = len(unique_values) <= 5
        high_rules = []  
        low_rules = []   
        
        if is_categorical:
            for threshold in unique_values:
                group = [row for row in raw_data if feature in row and row[feature] == threshold]
                if len(group) >= min_samples:
                    bullish_count = sum(1 for row in group if row.get(target_col) == 1)
                    bearish_count = sum(1 for row in group if row.get(target_col) == 0)
                    prob_bullish = bullish_count / len(group)
                    prob_bearish = bearish_count / len(group)
                    if prob_bullish > 0:
                        high_rules.append({"threshold": threshold, "operation": "==", "probability": prob_bullish, "support": len(group)})
                    if prob_bearish > 0:
                        low_rules.append({"threshold": threshold, "operation": "==", "probability": prob_bearish, "support": len(group)})
        else:
            candidate_bullish = []  
            candidate_bearish = []  
            for threshold in unique_values:
                group_lte = [row for row in raw_data if feature in row and row[feature] <= threshold]
                group_gt = [row for row in raw_data if feature in row and row[feature] > threshold]
                if len(group_lte) >= min_samples:
                    bullish_count = sum(1 for row in group_lte if row.get(target_col) == 1)
                    bearish_count = sum(1 for row in group_lte if row.get(target_col) == 0)
                    prob_bullish = bullish_count / len(group_lte)
                    prob_bearish = bearish_count / len(group_lte)
                    if prob_bullish > 0:
                        candidate_bullish.append({"threshold": threshold, "operation": "<=", "probability": prob_bullish, "support": len(group_lte)})
                    if prob_bearish > 0:
                        candidate_bearish.append({"threshold": threshold, "operation": "<=", "probability": prob_bearish, "support": len(group_lte)})
                
                if len(group_gt) >= min_samples:
                    bullish_count = sum(1 for row in group_gt if row.get(target_col) == 1)
                    bearish_count = sum(1 for row in group_gt if row.get(target_col) == 0)
                    prob_bullish = bullish_count / len(group_gt)
                    prob_bearish = bearish_count / len(group_gt)
                    if prob_bullish > 0:
                        candidate_bullish.append({"threshold": threshold, "operation": ">", "probability": prob_bullish, "support": len(group_gt)})
                    if prob_bearish > 0:
                        candidate_bearish.append({"threshold": threshold, "operation": ">", "probability": prob_bearish, "support": len(group_gt)})
            
            candidate_bullish.sort(key=lambda x: x["probability"], reverse=True)
            candidate_bearish.sort(key=lambda x: x["probability"], reverse=True)
            if candidate_bullish:
                top_5_bullish = candidate_bullish[:5]
                avg_threshold_bullish = sum(r["threshold"] for r in top_5_bullish) / len(top_5_bullish)
                high_rules = [{"threshold": avg_threshold_bullish, "operation": top_5_bullish[0]["operation"], "probability": top_5_bullish[0]["probability"], "support": top_5_bullish[0]["support"]}]
            if candidate_bearish:
                top_5_bearish = candidate_bearish[:5]
                avg_threshold_bearish = sum(r["threshold"] for r in top_5_bearish) / len(top_5_bearish)
                low_rules = [{"threshold": avg_threshold_bearish, "operation": top_5_bearish[0]["operation"], "probability": top_5_bearish[0]["probability"], "support": top_5_bearish[0]["support"]}]
        
        if high_rules or low_rules:
            results[feature] = {
                "high": [{"threshold": r["threshold"], "operation": r["operation"]} for r in high_rules],
                "low": [{"threshold": r["threshold"], "operation": r["operation"]} for r in low_rules]
            }
            
    return results

def select_combinations(features, combination_size):
    """
    Generate all unique combinations of features of a specified size using recursive algorithm.
    
    This function creates all possible feature combinations without replacement. For example,
    if features=["A", "B", "C"] and combination_size=2, it returns [["A", "B"], ["A", "C"], ["B", "C"]].
    This is used to create multi-feature trading rules that combine conditions across multiple features.
    
    Algorithm: Recursive approach that:
    1. Takes the first feature and combines it with all combinations of remaining features
    2. Recurses on the remaining features for subsequent combinations
    3. Base case: when combination_size is 0, returns [[]] (one empty combination)
    
    Parameters:
    - features: List of feature names (strings) to combine
    - combination_size: Number of features in each combination (integer)
    
    Returns:
    - List of feature combinations (list of lists)
    - Returns empty list if combination_size > len(features)
    """
    if combination_size > len(features):
        return []
    if combination_size == 0:
        return [[]]
    combinations = []
    for i in range(len(features)):
        current_feature = features[i]
        remaining_features = features[i + 1:]
        for sub_combination in select_combinations(remaining_features, combination_size - 1):
            combinations.append([current_feature] + sub_combination)
    return combinations

def generate_value_combinations(values):
    """
    Generate all combinations of threshold values from lists of values for each feature.
    
    This function creates the Cartesian product of value lists. Given lists of possible thresholds
    for each feature, it generates all possible combinations pairing one value from each list.
    For example, if values=[[0.5, 1.0], [10, 20]], returns [[0.5, 10], [0.5, 20], [1.0, 10], [1.0, 20]].
    This is used to evaluate all possible rule combinations across multiple features.
    
    Algorithm: Recursive approach that:
    1. Takes values from the first feature list
    2. Pairs each value with all sub-combinations from remaining feature lists
    3. Recurses on remaining feature lists
    4. Base case: when values list is empty, returns [[]] (one empty combination)
    
    Parameters:
    - values: List of lists, where each inner list contains threshold values for a feature
    
    Returns:
    - List of value combinations (list of lists)
    - Returns [[]] if input is empty list
    """
    combinations = []
    if len(values) == 0:
        return [[]]
    current_values_list = values[0]
    remaining_values_list = values[1:]
    for i in range(len(values[0])):
        current_value = current_values_list[i]
        for sub_combination in generate_value_combinations(remaining_values_list):
            combinations.append([current_value] + sub_combination)
    return combinations

def evaluate_combination_set(raw_data, combination_features, rule_combinations, is_high_target, min_support, target_col, pct_col, day_col):
    """
    Evaluate all rule combinations and find the one with the highest win rate for a feature set.
    
    This function tests multiple rule combinations (different thresholds/operators for the same features)
    against the training data to find which combination produces the best trading signals. For each
    combination, it counts how many data points match all rules, calculates the win rate (percentage
    of matching points with target=1 or target=0), and tracks associated profit metrics.
    
    Process:
    1. For each rule combination (set of thresholds/operators for the features):
       - Count how many data rows match ALL rules simultaneously
       - Count wins (target=1 for high, target=0 for low)
       - Calculate win rate as percentage
       - Average profit percentage across matching rows
       - Average days until signal (if applicable)
    2. Keep track of the best performing combination
    3. Return the best combination's stats and rules
    
    Parameters:
    - raw_data: List of dictionaries with feature columns and target information
    - combination_features: List of feature names being analyzed (strings)
    - rule_combinations: List of rule combinations to evaluate, each rule combination is a list of
                        dicts with "threshold" and "operation" keys
    - is_high_target: Boolean indicating if we're optimizing for HIGH (1) or LOW (0) targets
    - min_support: Minimum number of matching rows required for a valid combination (integer)
    - target_col: Name of the target column (string) containing 0/1 values
    - pct_col: Name of the percentage profit column (string) to average
    - day_col: Name of the day column to average for signal timing (string or None if not used)
    
    Returns:
    - Dictionary with best combination containing:
      - "win_rate": Percentage of matching rows that hit the target (float 0-1)
      - "support": Number of matching rows (integer)
      - "profit": Average percentage profit across matching rows (float)
      - "which_day": Average days to signal (float or None)
      - "rules": List of rule dicts with "threshold" and "operation" for each feature
    - Returns None if no combination meets the min_support requirement
    """
    best_percentage = -1
    best_rules = None
    best_support = 0
    best_profit = 0.0
    best_day = None

    for rule_combination in rule_combinations:
        match_count = 0
        ones_count = 0
        sum_pct = 0.0
        sum_day = 0.0
        day_count = 0

        for row in raw_data:
            row_include = True
            for k in range(len(combination_features)):
                val = row.get(combination_features[k])
                rule = rule_combination[k]
                if val is None or not OPS[rule["operation"]](val, rule["threshold"]):
                    row_include = False
                    break
                    
            if row_include:
                target_val = row.get(target_col)
                if target_val is not None:
                    match_count += 1
                    if target_val == 1:
                        ones_count += 1
                    
                    pct_val = row.get(pct_col)
                    if pct_val is not None:
                        sum_pct += pct_val
                        
                    if day_col:
                        d_val = row.get(day_col)
                        if d_val is not None:
                            sum_day += d_val
                            day_count += 1
        
        if match_count >= min_support:
            ones_percentage = ones_count / match_count
            target_percentage = ones_percentage if is_high_target else (1.0 - ones_percentage)
            
            if target_percentage > best_percentage:
                best_percentage = target_percentage
                best_rules = rule_combination
                best_support = match_count
                best_profit = sum_pct / match_count if match_count > 0 else 0.0
                best_day = sum_day / day_count if day_count > 0 else None

    if best_rules is not None:
        return {
            "win_rate": best_percentage,
            "support": best_support,
            "profit": best_profit,
            "which_day": best_day,
            "rules": best_rules
        }
    return None

def engine():
    """
    Main execution function that discovers optimal trading rules through exhaustive feature combination analysis.
    
    This is the orchestrating function that coordinates the entire rule discovery pipeline. It:
    1. Loads historical market data with technical indicators from final_matrix.json
    2. Splits data into training (80%) and test sets
    3. For each target variable (3-day, 90-day, 180-day, 365-day outlook):
       - Finds optimal single-feature thresholds
       - Tests all possible multi-feature combinations (2 features, 3 features, etc.)
       - For each combination, generates bullish (HIGH) and bearish (LOW) trading rules
       - Tracks win rates, support counts, and profit metrics
    4. Saves discovered rules to JSON files organized by target and direction
    
    The engine discovers rules by:
    - Testing categorical conditions (exact equality) for features with few unique values
    - Testing continuous conditions (<=, >) for features with many unique values
    - Evaluating all combinations of rules to find the most predictive multi-feature signals
    - Filtering out rules with insufficient historical support (minimum 10 occurrences)
    
    User Interaction:
    - Prompts for ticker symbol (stock symbol like 'aapl')
    - Reads from: final_matrix.json (generated by matrix_creator.py)
    - Outputs: {ticker}_{target}.json files containing discovered rules
      (e.g., aapl_Target.json, aapl_Target_90.json, etc.)
    
    Each output file contains JSON objects with structure:
    - Key: Rule identifier (e.g., "HIGH-Direction-SMA_20_Ratio-RSI")
    - Value: Object with day, type, combination, win_rate, support, profit, rules
    """
    print("================================ Gini Engine Master =================================")
    ticker = input("Hisse adini giriniz (orn. aapl): ").strip().lower()
    
    with open("final_matrix.json", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    X_train, X_test, y_train, y_test = test_creator.create_test_data(raw_data)
    train_size = int(len(raw_data) * 0.8)
    train_data = raw_data[:train_size]
    
    MIN_SUPPORT = 10 

    for target_col, config in TARGET_CONFIG.items():
        print(f"\n[{target_col}] Test Ediliyor...")
        best_thresholds = get_best_single_thresholds(train_data, features_to_keep, target_col)
        final_best_combinations = {}

        for i in range(1, len(features_to_keep)):
            combinations = select_combinations(features_to_keep, i + 1)
            for combination in combinations:
                
                high_values = [best_thresholds.get(feat, {}).get("high", []) for feat in combination]
                low_values = [best_thresholds.get(feat, {}).get("low", []) for feat in combination]

                value_combinations_high = generate_value_combinations(high_values)
                best_high = evaluate_combination_set(train_data, combination, value_combinations_high, True, MIN_SUPPORT, target_col, config["pct_col"], config["day_col"])
                
                if best_high:
                    rule_objects_high = []
                    for k in range(len(combination)):
                        rule_objects_high.append(Rule(feature=combination[k], threshold=best_high["rules"][k]["threshold"], operation=best_high["rules"][k]["operation"]))
                        
                    val_high_obj = Value(
                        day=config["days"],
                        type=Type.HIGH,
                        combination=combination,
                        win_rate=best_high["win_rate"],
                        support=best_high["support"],
                        percentage_profit=best_high["profit"],
                        which_day_is_higher=best_high["which_day"],
                        rules=rule_objects_high
                    )
                    key_name = f"HIGH-{'-'.join(combination)}"
                    final_best_combinations[key_name] = val_high_obj.to_dict()

                value_combinations_low = generate_value_combinations(low_values)
                best_low = evaluate_combination_set(train_data, combination, value_combinations_low, False, MIN_SUPPORT, target_col, config["pct_col"], config["day_col"])
                
                if best_low:
                    rule_objects_low = []
                    for k in range(len(combination)):
                        rule_objects_low.append(Rule(feature=combination[k], threshold=best_low["rules"][k]["threshold"], operation=best_low["rules"][k]["operation"]))
                        
                    val_low_obj = Value(
                        day=config["days"],
                        type=Type.LOW,
                        combination=combination,
                        win_rate=best_low["win_rate"],
                        support=best_low["support"],
                        percentage_profit=best_low["profit"],
                        which_day_is_higher=best_low["which_day"],
                        rules=rule_objects_low
                    )
                    key_name = f"LOW-{'-'.join(combination)}"
                    final_best_combinations[key_name] = val_low_obj.to_dict()

        output_filename = f"{ticker}_{target_col}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_best_combinations, f, indent=4)
        print(f"[{target_col}] Sonuclari {output_filename} icerisine obje formatinda kaydedildi.")

if __name__ == "__main__":
    engine()
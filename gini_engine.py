import test_creator
import json
import operator

features_to_keep = ["Direction", "SMA_20_Ratio", "SMA_50_Ratio", "SMA_100_Ratio", "SMA_200_Ratio", "Cross_20_50", "Cross_50_100", "Cross_100_200", "MACD_Line", "MACD_Signal", "RSI"]

OPS = {
    "==": operator.eq,
    "<=": operator.le,
    ">=": operator.ge,
    "<":  operator.lt,
    ">":  operator.gt
}

def get_best_single_thresholds(raw_data, features_to_keep, target_col, min_samples=100):
    """
    Find the best threshold values for each feature that maximize the probability of bullish or bearish signals.
    For categorical features, evaluates each unique value. For continuous features, evaluates LTE and GT operations.
    Returns a dictionary mapping feature names to their high and low rules.
    """
    results = {}
    
    for feature in features_to_keep:
        feature_values = [
            row[feature] for row in raw_data 
            if feature in row and row[feature] is not None
        ]
        
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
    Generate all combinations of features with a given size using recursive approach.
    Returns a list of all unique combinations of the specified size from the input features.
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
    Generate all combinations of values from multiple lists using recursive approach.
    Takes a list of lists and returns all possible combinations taking one value from each list.
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

def calculate_one_percentage(ones_list):
    """
    Calculate the percentage of values equal to 1 in the given list.
    Returns the ratio of ones to total count.
    """
    ones = sum(1 for number in ones_list if number == 1)
    return ones / len(ones_list)

def evaluate_combination_set(raw_data, combination_features, rule_combinations, is_high_target, min_support, target_col):
    """
    Evaluate a set of rule combinations against raw data to find the best performing rules.
    Filters data by rules, calculates win rates based on target variable, and returns the best combination.
    Only considers combinations that meet the minimum support threshold.
    """
    best_percentage = -1
    best_rules = None
    best_support = 0

    for rule_combination in rule_combinations:
        filtered_targets = []
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
                    filtered_targets.append(target_val)
        
        support = len(filtered_targets)
        if support >= min_support:
            ones_percentage = calculate_one_percentage(filtered_targets)
            target_percentage = ones_percentage if is_high_target else (1.0 - ones_percentage)
            
            if target_percentage > best_percentage:
                best_percentage = target_percentage
                best_rules = rule_combination
                best_support = support

    if best_rules is not None:
        return {
            "win_rate": best_percentage,
            "support": best_support,
            "rules": best_rules
        }
    return None

def engine():
    """
    Main engine function that generates trading rules by testing feature combinations.
    Loads final matrix data, generates best single thresholds, and evaluates multi-feature combinations.
    Saves the best combinations with their win rates and support to best_combinations.json.
    """
    print("================================ Gini Engine =================================")
    ticker = input("Hisse adini giriniz (orn. aapl): ").strip().lower()
    
    with open("final_matrix.json", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    X_train, X_test, y_train, y_test = test_creator.create_test_data(raw_data)
    
    train_size = int(len(raw_data) * 0.8)
    train_data = raw_data[:train_size]
    
    targets = ["Target", "Target_90", "Target_180", "Target_365"]
    MIN_SUPPORT = 10 

    for target_col in targets:
        best_thresholds = get_best_single_thresholds(train_data, features_to_keep, target_col)
        final_best_combinations = {}

        for i in range(1, len(features_to_keep)):
            combinations = select_combinations(features_to_keep, i + 1)
            for combination in combinations:
                print(f"Testing combination for size {i + 1}: {combination} on {target_col}")
                
                high_values = [best_thresholds.get(feat, {}).get("high", []) for feat in combination]
                low_values = [best_thresholds.get(feat, {}).get("low", []) for feat in combination]

                value_combinations_high = generate_value_combinations(high_values)
                best_high = evaluate_combination_set(train_data, combination, value_combinations_high, True, MIN_SUPPORT, target_col)
                
                if best_high:
                    key_name = f"HIGH-{'-'.join(combination)}"
                    final_best_combinations[key_name] = best_high

                value_combinations_low = generate_value_combinations(low_values)
                best_low = evaluate_combination_set(train_data, combination, value_combinations_low, False, MIN_SUPPORT, target_col)
                
                if best_low:
                    key_name = f"LOW-{'-'.join(combination)}"
                    final_best_combinations[key_name] = best_low

        output_filename = f"{ticker}_{target_col}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_best_combinations, f, indent=4)

if __name__ == "__main__":
    engine()
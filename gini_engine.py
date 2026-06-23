import test_creator
import json

features_to_keep = ["Direction", "SMA_20_Ratio", "SMA_50_Ratio", "SMA_100_Ratio", "SMA_200_Ratio", "Cross_20_50", "Cross_50_100", "Cross_100_200", "MACD_Line", "MACD_Signal", "RSI"]


def get_best_single_thresholds(raw_data, features_to_keep, min_samples=100):
    """
    Greedy search function to find best single-feature rules predicting Bullish (Target=1) 
    and Bearish (Target=0) movements.
    
    For continuous features (>5 unique values):
    - Finds top 5 best rules for 'High' and top 5 for 'Low'
    - Averages their thresholds to create ONE robust "Champion Threshold"
    - Returns only this single averaged rule
    
    For categorical/discrete features (≤5 unique values):
    - Uses strict equality (==) matching
    - Handles features like Direction (-1, 0, 1) and Cross signals (0, 1) natively
    - Returns all valid rules without averaging
    
    Args:
        raw_data: List of dictionaries, each with feature columns and a 'Target' key (0 or 1)
        features_to_keep: List of feature names to evaluate
        min_samples: Minimum samples required per group to consider a rule (default: 100)
    
    Returns:
        Nested dictionary mapping features to their best rules:
        {
            "feature_name": {
                "high": [{"threshold": X, "operation": "op"}],
                "low": [{"threshold": X, "operation": "op"}]
            },
            ...
        }
    """
    results = {}
    
    for feature in features_to_keep:
        # Extract all non-None values for this feature
        feature_values = [
            row[feature] for row in raw_data 
            if feature in row and row[feature] is not None
        ]
        
        if not feature_values:
            # Skip features with no valid data
            continue
        
        # Determine feature type: categorical (≤5 unique values) vs continuous (>5 unique values)
        unique_values = sorted(set(feature_values))
        is_categorical = len(unique_values) <= 5
        
        high_rules = []  # Rules predicting Target == 1 (Bullish)
        low_rules = []   # Rules predicting Target == 0 (Bearish)
        
        if is_categorical:
            # CATEGORICAL/DISCRETE FEATURES (e.g., Direction: -1, 0, 1 or Cross signals: 0, 1)
            # Split by strict equality (==) rather than comparison (<= and >)
            # Return all valid rules without averaging
            
            for threshold in unique_values:
                # Group: rows where feature == threshold
                group = [
                    row for row in raw_data 
                    if feature in row and row[feature] == threshold
                ]
                
                # Apply minimum support constraint
                if len(group) >= min_samples:
                    # Calculate probability of Target == 1 (bullish) in this group
                    bullish_count = sum(1 for row in group if row.get("Target") == 1)
                    bearish_count = sum(1 for row in group if row.get("Target") == 0)
                    
                    prob_bullish = bullish_count / len(group)
                    prob_bearish = bearish_count / len(group)
                    
                    # Add bullish rule if probability > 0
                    if prob_bullish > 0:
                        high_rules.append({
                            "threshold": threshold,
                            "operation": "==",
                            "probability": prob_bullish,
                            "support": len(group)
                        })
                    
                    # Add bearish rule if probability > 0
                    if prob_bearish > 0:
                        low_rules.append({
                            "threshold": threshold,
                            "operation": "==",
                            "probability": prob_bearish,
                            "support": len(group)
                        })
        
        else:
            # CONTINUOUS FEATURES (e.g., RSI, MACD, SMA_Ratio values)
            # Split by threshold comparison (<= and >)
            # Find top 5 rules, average their thresholds, return ONE champion threshold
            
            candidate_bullish = []  # All rules with bullish predictive power
            candidate_bearish = []  # All rules with bearish predictive power
            
            for threshold in unique_values:
                # Group A: rows where feature <= threshold
                group_lte = [
                    row for row in raw_data 
                    if feature in row and row[feature] <= threshold
                ]
                
                # Group B: rows where feature > threshold
                group_gt = [
                    row for row in raw_data 
                    if feature in row and row[feature] > threshold
                ]
                
                # Evaluate Group A (feature <= threshold)
                if len(group_lte) >= min_samples:
                    bullish_count = sum(1 for row in group_lte if row.get("Target") == 1)
                    bearish_count = sum(1 for row in group_lte if row.get("Target") == 0)
                    
                    prob_bullish = bullish_count / len(group_lte)
                    prob_bearish = bearish_count / len(group_lte)
                    
                    if prob_bullish > 0:
                        candidate_bullish.append({
                            "threshold": threshold,
                            "operation": "<=",
                            "probability": prob_bullish,
                            "support": len(group_lte)
                        })
                    
                    if prob_bearish > 0:
                        candidate_bearish.append({
                            "threshold": threshold,
                            "operation": "<=",
                            "probability": prob_bearish,
                            "support": len(group_lte)
                        })
                
                # Evaluate Group B (feature > threshold)
                if len(group_gt) >= min_samples:
                    bullish_count = sum(1 for row in group_gt if row.get("Target") == 1)
                    bearish_count = sum(1 for row in group_gt if row.get("Target") == 0)
                    
                    prob_bullish = bullish_count / len(group_gt)
                    prob_bearish = bearish_count / len(group_gt)
                    
                    if prob_bullish > 0:
                        candidate_bullish.append({
                            "threshold": threshold,
                            "operation": ">",
                            "probability": prob_bullish,
                            "support": len(group_gt)
                        })
                    
                    if prob_bearish > 0:
                        candidate_bearish.append({
                            "threshold": threshold,
                            "operation": ">",
                            "probability": prob_bearish,
                            "support": len(group_gt)
                        })
            
            # Sort by probability (descending)
            candidate_bullish.sort(key=lambda x: x["probability"], reverse=True)
            candidate_bearish.sort(key=lambda x: x["probability"], reverse=True)
            
            # Extract top 5, average their thresholds, create ONE champion threshold for each
            if candidate_bullish:
                top_5_bullish = candidate_bullish[:5]
                avg_threshold_bullish = sum(r["threshold"] for r in top_5_bullish) / len(top_5_bullish)
                champion_operation_bullish = top_5_bullish[0]["operation"]  # Operation from #1 best rule
                high_rules = [{
                    "threshold": avg_threshold_bullish,
                    "operation": champion_operation_bullish,
                    "probability": top_5_bullish[0]["probability"],
                    "support": top_5_bullish[0]["support"]
                }]
            
            if candidate_bearish:
                top_5_bearish = candidate_bearish[:5]
                avg_threshold_bearish = sum(r["threshold"] for r in top_5_bearish) / len(top_5_bearish)
                champion_operation_bearish = top_5_bearish[0]["operation"]  # Operation from #1 best rule
                low_rules = [{
                    "threshold": avg_threshold_bearish,
                    "operation": champion_operation_bearish,
                    "probability": top_5_bearish[0]["probability"],
                    "support": top_5_bearish[0]["support"]
                }]
        
        # Add to results if we found any valid rules for this feature
        if high_rules or low_rules:
            # Strip probability and support, keeping only threshold and operation
            high_rules_simplified = [{"threshold": r["threshold"], "operation": r["operation"]} for r in high_rules]
            low_rules_simplified = [{"threshold": r["threshold"], "operation": r["operation"]} for r in low_rules]
            
            results[feature] = {
                "high": high_rules_simplified,
                "low": low_rules_simplified
            }
    
    return results


def select_combinations(features, combination_size):
    if combination_size > len(features):
        return []
    if combination_size == 0:
        return [[]]
    combinations = []
    for i in range(len(features)):
        current_feature = features[i]
        remaining_features = features[i + 1:]
        sub_combinations = select_combinations(remaining_features, combination_size - 1)
        for sub_combination in sub_combinations:
            combinations.append([current_feature] + sub_combination)
    return combinations


def generate_value_combinations(values):
    combinations = []
    if len(values) == 0:
        return [[]]
    current_values_list = values[0]
    remaining_values_list = values[1:]
    for i in range(len(values[0])):
        current_value = current_values_list[i]
        sub_combinations = generate_value_combinations(remaining_values_list)
        for sub_combination in sub_combinations:
            combinations.append([current_value] + sub_combination)
    return combinations
        

def engine():
    print("================================ Gini Engine =================================")
    with open("final_matrix.json", encoding="utf-8") as f:
        raw_data = json.load(f)
    # X_train, X_test, y_train, y_test = test_creator.create_test_data(raw_data)
    best_thresholds = get_best_single_thresholds(raw_data, features_to_keep)
    #print(json.dumps(best_thresholds, indent=4))
    with open("value_combinations.txt", "a", encoding="utf-8") as f:
        for i in range(1, len(features_to_keep)):
            combinations = select_combinations(features_to_keep, i + 1)
            for combination in combinations:
                #with open("combinations.txt", "a", encoding="utf-8") as f:
                #f.write(f"{combination}\n")
                print(f"Testing combination for size {i + 1}: {combination}")
                values_dict = {
                    "high": [],
                    "low": []
                }
                for j in range(len(combination)):
                    feature = combination[j]
                    values = []
                    for value in best_thresholds[feature]["high"]:
                        values.append(value)
                    values_dict["high"].append(values)
                    values = []
                    for value in best_thresholds[feature]["low"]:
                        values.append(value)
                    values_dict["low"].append(values)

                value_combinations = generate_value_combinations(values_dict["high"])
                for value_combination in value_combinations:
                    f.write(f"{combination}: {value_combination}\n")
                
                value_combinations = generate_value_combinations(values_dict["low"])
                for value_combination in value_combinations:
                    f.write(f"{combination}: {value_combination}\n")

                


if __name__ == "__main__":
    engine()
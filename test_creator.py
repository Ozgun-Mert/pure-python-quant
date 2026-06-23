import json

TRAIN_SIZE = 0.8

def create_test_data(matrix, day = "Target"):
    """
    Split the matrix into training and test datasets while extracting features and targets.
    Uses 80% of data for training and 20% for testing. Returns feature and target arrays for both sets.
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    total_rows = len(matrix)
    split_index = int(total_rows * TRAIN_SIZE)
    train_data = matrix[:split_index]
    test_data = matrix[split_index:]

    features_to_keep = ["Direction", "SMA_20_Ratio", "SMA_50_Ratio", "SMA_100_Ratio", "SMA_200_Ratio", "Cross_20_50", "Cross_50_100", "Cross_100_200", "MACD_Line", "MACD_Signal", "RSI"]
    def extract_features(data):
        """
        Extract feature vectors and target values from data rows.
        Returns two lists: X (feature vectors) and y (target values).
        """
        X = []
        y = []

        for row in data:
            x_row = [row[feature] for feature in features_to_keep]
            X.append(x_row)
            y.append(row[day])
            
        return X, y

    X_train, y_train = extract_features(train_data)
    X_test, y_test = extract_features(test_data)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    with open("final_matrix.json", encoding="utf-8") as f:
        raw_data = json.load(f)
    X_train, X_test, y_train, y_test = create_test_data(raw_data)
    print(f"X_train: {len(X_train)}")
    print(f"X_test: {len(X_test)}")
    print(f"y_train: {len(y_train)}")
    print(f"y_test: {len(y_test)}")
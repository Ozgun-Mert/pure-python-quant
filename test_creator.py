"""
Train/test data splitting module for the Pure Python Quant pipeline.

Provides utilities to split the feature matrix into training and test sets
for model evaluation. Uses a chronological 80/20 split (no shuffling) to
respect the time-series nature of financial data and avoid look-ahead bias.

The create_test_data() function extracts feature vectors and binary target
labels from matrix rows, returning four arrays suitable for sklearn-style
model training and evaluation workflows.
"""

import json

TRAIN_SIZE = 0.8
"""
Fraction of the matrix reserved for training data.

The remaining (1 - TRAIN_SIZE) portion becomes the test set. Applied as a
chronological front/back split — the earliest 80% of rows train, the latest
20% test — preserving temporal ordering.
"""

def create_test_data(matrix, day = "Target"):
    """
    Split the feature matrix into chronological training and test datasets.

    Divides the input matrix at the TRAIN_SIZE boundary (80% train, 20% test),
    then extracts feature vectors from eleven technical indicators and the
    corresponding binary target label for each row.

    Features extracted (in order):
        Direction, SMA_20_Ratio, SMA_50_Ratio, SMA_100_Ratio, SMA_200_Ratio,
        Cross_20_50, Cross_50_100, Cross_100_200, MACD_Line, MACD_Signal, RSI

    Parameters:
        matrix: List of row dictionaries from matrix.json, each containing
                feature columns and target columns.
        day: Name of the target column to use as the label (default: "Target").
             Can be "Target", "Target_90", "Target_180", or "Target_365".

    Returns:
        tuple: (X_train, X_test, y_train, y_test) where X arrays are lists of
               feature vectors (list of floats) and y arrays are lists of
               binary target values (0 or 1, possibly None for recent rows).
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
        Extract parallel feature vectors and target labels from a data subset.

        For each row in the input data, builds a feature vector by collecting
        values for all features_to_keep columns in a fixed order, and appends
        the row's target label from the specified day column.

        Parameters:
            data: Subset of matrix rows (train or test partition).

        Returns:
            tuple: (X, y) where X is a list of feature vectors and y is a list
                   of corresponding target values.
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
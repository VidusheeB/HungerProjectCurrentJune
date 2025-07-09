import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends.csv"
MODELS_DIR = "county_models"


def load_model():
    with open(os.path.join(MODELS_DIR, "global_model.pkl"), "rb") as f:
        return pickle.load(f)

def prepare_data():
    df = pd.read_csv(AGGREGATE_TRENDS_FILE)
    feature_cols = ["Population"] + [col for col in df.columns if col.startswith('monthly_average_')]
    X = df[feature_cols]
    y = df["SNAP_Applications"]
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    return X, y, feature_cols

def run_accuracy_tests():
    print("=== LINEAR REGRESSION MODEL ACCURACY ASSESSMENT ===\n")
    X, y, feature_cols = prepare_data()
    model_info = load_model()
    model = model_info["model"]
    # 1. Full dataset
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    print(f"Full Data MAE: {mae:.2f}")
    print(f"Full Data RMSE: {rmse:.2f}")
    print(f"Full Data R²: {r2:.4f}")
    print(f"Full Data MAPE: {mape:.2f}%")
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_model = LinearRegression().fit(X_train, y_train)
    y_test_pred = train_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAPE: {test_mape:.2f}%")
    # 3. Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"CV R² Scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    return mae, rmse, mape, test_mae, test_rmse, test_mape, cv_scores

if __name__ == "__main__":
    run_accuracy_tests() 
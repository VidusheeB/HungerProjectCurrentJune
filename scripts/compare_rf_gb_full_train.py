import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends.csv"

def prepare_data():
    df = pd.read_csv(AGGREGATE_TRENDS_FILE)
    feature_cols = ["Population"] + [col for col in df.columns if col.startswith('monthly_average_')]
    X = df[feature_cols]
    y = df["SNAP_Applications"]
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    return X, y, feature_cols

def report_metrics(y_true, y_pred, model, feature_cols):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    return mae, rmse, r2, mape, feature_importance

def main():
    print("=== FULL DATASET TRAINING: RANDOM FOREST vs GRADIENT BOOSTING ===\n")
    X, y, feature_cols = prepare_data()
    print(f"Total data points: {len(X)}")
    print(f"Features: {feature_cols}")
    print(f"Target variable: SNAP_Applications\n")

    # Random Forest
    print("Training RandomForestRegressor on full data...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)
    rf_mae, rf_rmse, rf_r2, rf_mape, rf_importance = report_metrics(y, y_pred_rf, rf, feature_cols)

    # Gradient Boosting
    print("\nTraining GradientBoostingRegressor on full data...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X, y)
    y_pred_gb = gb.predict(X)
    gb_mae, gb_rmse, gb_r2, gb_mape, gb_importance = report_metrics(y, y_pred_gb, gb, feature_cols)

    print("\n=== RANDOM FOREST REGRESSOR (FULL DATA) ===")
    print(f"MAE: {rf_mae:.2f}")
    print(f"RMSE: {rf_rmse:.2f}")
    print(f"R²: {rf_r2:.4f}")
    print(f"MAPE: {rf_mape:.2f}%")
    print("Feature importances:")
    print(rf_importance.to_string(index=False))

    print("\n=== GRADIENT BOOSTING REGRESSOR (FULL DATA) ===")
    print(f"MAE: {gb_mae:.2f}")
    print(f"RMSE: {gb_rmse:.2f}")
    print(f"R²: {gb_r2:.4f}")
    print(f"MAPE: {gb_mape:.2f}%")
    print("Feature importances:")
    print(gb_importance.to_string(index=False))

    print("\nNote: Tree-based models (Random Forest, Gradient Boosting) do not provide p-values for feature significance. Use feature importances for interpretation.")

    # Save results
    results = {
        'rf_mae': rf_mae,
        'rf_rmse': rf_rmse,
        'rf_r2': rf_r2,
        'rf_mape': rf_mape,
        'gb_mae': gb_mae,
        'gb_rmse': gb_rmse,
        'gb_r2': gb_r2,
        'gb_mape': gb_mape,
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv("src/data/compare_rf_gb_full_train_results.csv", index=False)
    print("\nDetailed results saved to: src/data/compare_rf_gb_full_train_results.csv")

if __name__ == "__main__":
    main() 
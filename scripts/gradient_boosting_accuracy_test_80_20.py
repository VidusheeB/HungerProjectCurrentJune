import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

def run_gb_80_20_test():
    print("=== 80-20 TRAIN-TEST SPLIT: GRADIENT BOOSTING REGRESSOR ===")
    print("Using linear interpolated data\n")
    X, y, feature_cols = prepare_data()
    print(f"Total data points: {len(X)}")
    print(f"Features: {feature_cols}")
    print(f"Target variable: SNAP_Applications")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set size: {len(X_train)} (80%)")
    print(f"Test set size: {len(X_test)} (20%)")
    print("\nTraining GradientBoostingRegressor on 80% of data...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_train = gb.predict(X_train)
    y_pred_test = gb.predict(X_test)
    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    # Print results
    print("\n=== TRAIN SET PERFORMANCE (80% of data) ===")
    print(f"MAE: {train_mae:.2f}")
    print(f"RMSE: {train_rmse:.2f}")
    print(f"R²: {train_r2:.4f}")
    print(f"MAPE: {train_mape:.2f}%")
    print("\n=== TEST SET PERFORMANCE (20% of data) ===")
    print(f"MAE: {test_mae:.2f}")
    print(f"RMSE: {test_rmse:.2f}")
    print(f"R²: {test_r2:.4f}")
    print(f"MAPE: {test_mape:.2f}%")
    print("\n=== FEATURE IMPORTANCE ===")
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': gb.feature_importances_
    }).sort_values('Importance', ascending=False)
    for idx, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    print("\n=== SAMPLE PREDICTIONS ANALYSIS ===")
    sample_size = min(10, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    print(f"Sample of {sample_size} test predictions:")
    print("Actual\t\tPredicted\t\tDifference\t\t% Error")
    print("-" * 70)
    for idx in sample_indices:
        actual = y_test.iloc[idx]
        predicted = y_pred_test[idx]
        diff = predicted - actual
        pct_error = (abs(diff) / actual) * 100 if actual != 0 else 0
        print(f"{actual:.0f}\t\t{predicted:.0f}\t\t{diff:+.0f}\t\t{pct_error:.1f}%")
    # Save results
    results = {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_mape': train_mape,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'feature_importance': feature_importance.to_dict('records')
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv("src/data/gradient_boosting_80_20_test_results.csv", index=False)
    print(f"\nDetailed results saved to: src/data/gradient_boosting_80_20_test_results.csv")
    return results

if __name__ == "__main__":
    try:
        results = run_gb_80_20_test()
        print("\n✅ 80-20 Gradient Boosting regression test completed successfully!")
    except Exception as e:
        print(f"❌ Error during Gradient Boosting 80-20 test: {str(e)}")
        import traceback
        traceback.print_exc() 
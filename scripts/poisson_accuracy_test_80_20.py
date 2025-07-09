import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
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

def pseudo_r2(y_true, y_pred):
    # McFadden's pseudo R^2
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    llf = np.sum(y_true * np.log(y_pred + 1e-9) - y_pred)
    llnull = np.sum(y_true * np.log(np.mean(y_true) + 1e-9) - np.mean(y_true))
    return 1 - llf / llnull

def run_poisson_80_20_test():
    print("=== 80-20 TRAIN-TEST SPLIT: POISSON REGRESSION ===")
    print("Using PCHIP interpolated data\n")
    X, y, feature_cols = prepare_data()
    print(f"Total data points: {len(X)}")
    print(f"Features: {feature_cols}")
    print(f"Target variable: SNAP_Applications")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set size: {len(X_train)} (80%)")
    print(f"Test set size: {len(X_test)} (20%)")
    # Add constant for intercept
    X_train_c = sm.add_constant(X_train)
    X_test_c = sm.add_constant(X_test)
    print("\nTraining Poisson regression on 80% of data...")
    poisson_model = sm.GLM(y_train, X_train_c, family=sm.families.Poisson()).fit()
    y_pred_train = poisson_model.predict(X_train_c)
    y_pred_test = poisson_model.predict(X_test_c)
    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
    train_pseudo_r2 = pseudo_r2(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    test_pseudo_r2 = pseudo_r2(y_test, y_pred_test)
    # Print results
    print("\n=== TRAIN SET PERFORMANCE (80% of data) ===")
    print(f"MAE: {train_mae:.2f}")
    print(f"RMSE: {train_rmse:.2f}")
    print(f"Pseudo R²: {train_pseudo_r2:.4f}")
    print(f"MAPE: {train_mape:.2f}%")
    print("\n=== TEST SET PERFORMANCE (20% of data) ===")
    print(f"MAE: {test_mae:.2f}")
    print(f"RMSE: {test_rmse:.2f}")
    print(f"Pseudo R²: {test_pseudo_r2:.4f}")
    print(f"MAPE: {test_mape:.2f}%")
    print("\n=== FEATURE COEFFICIENTS ===")
    for name, coef in zip(['const'] + feature_cols, poisson_model.params):
        print(f"{name}: {coef:.6f}")
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
        'train_pseudo_r2': train_pseudo_r2,
        'train_mape': train_mape,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_pseudo_r2': test_pseudo_r2,
        'test_mape': test_mape,
        'feature_coefficients': dict(zip(['const'] + feature_cols, poisson_model.params))
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv("src/data/poisson_80_20_test_results.csv", index=False)
    print(f"\nDetailed results saved to: src/data/poisson_80_20_test_results.csv")
    return results

if __name__ == "__main__":
    try:
        results = run_poisson_80_20_test()
        print("\n✅ 80-20 Poisson regression test completed successfully!")
    except Exception as e:
        print(f"❌ Error during Poisson 80-20 test: {str(e)}")
        import traceback
        traceback.print_exc() 
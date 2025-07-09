import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends.csv"
MODELS_DIR = "county_models"

def load_model():
    """Load the trained global model."""
    with open(os.path.join(MODELS_DIR, "global_model.pkl"), "rb") as f:
        return pickle.load(f)

def prepare_data():
    """Prepare the data for training and testing."""
    df = pd.read_csv(AGGREGATE_TRENDS_FILE)
    feature_cols = ["Population"] + [col for col in df.columns if col.startswith('monthly_average_')]
    X = df[feature_cols]
    y = df["SNAP_Applications"]
    
    # Drop rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Ensure target values are non-negative
    y = y.clip(lower=0)
    
    return X, y, feature_cols

def run_80_20_test():
    """Run 80-20 train-test split accuracy assessment."""
    print("=== 80-20 TRAIN-TEST SPLIT ACCURACY ASSESSMENT ===")
    print("Using PCHIP interpolated data\n")
    
    # Prepare data
    X, y, feature_cols = prepare_data()
    print(f"Total data points: {len(X)}")
    print(f"Features: {feature_cols}")
    print(f"Target variable: SNAP_Applications")
    
    # Perform 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set size: {len(X_train)} (80%)")
    print(f"Test set size: {len(X_test)} (20%)")
    
    # Train Random Forest model on 80% of data (same as production)
    print("\nTraining model on 80% of data...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    # (Temporarily removed non-negative constraint for testing)
    # y_pred_test = np.maximum(0, y_pred_test)
    # y_pred_train = np.maximum(0, y_pred_train)
    
    # Calculate metrics for test set
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    # Calculate metrics for train set
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
    
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
    
    # Calculate overfitting metrics
    overfitting_mae = train_mae - test_mae
    overfitting_rmse = train_rmse - test_rmse
    overfitting_r2 = test_r2 - train_r2
    
    print("\n=== OVERFITTING ANALYSIS ===")
    print(f"MAE difference (Train - Test): {overfitting_mae:.2f}")
    print(f"RMSE difference (Train - Test): {overfitting_rmse:.2f}")
    print(f"R² difference (Test - Train): {overfitting_r2:.4f}")
    
    if overfitting_r2 < -0.1:
        print("⚠️  Potential overfitting detected (Test R² significantly lower than Train R²)")
    elif overfitting_r2 > 0.1:
        print("✅ Good generalization (Test R² higher than Train R²)")
    else:
        print("✅ Model shows reasonable generalization")
    
    # Feature importance analysis
    print("\n=== FEATURE IMPORTANCE ===")
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.feature_importances_
    })
    feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Coefficient']:.6f}")
    
    # Sample predictions analysis
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
    
    # Check for negative predictions
    negative_predictions = np.sum(y_pred_test < 0)
    if negative_predictions > 0:
        print(f"\n⚠️  Warning: {negative_predictions} negative predictions detected")
    else:
        print(f"\n✅ All predictions are non-negative (as expected)")
    
    # Save results to file
    results = {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'train_mape': train_mape,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'overfitting_mae': overfitting_mae,
        'overfitting_rmse': overfitting_rmse,
        'overfitting_r2': overfitting_r2,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    # Save detailed results
    results_df = pd.DataFrame([results])
    results_df.to_csv("src/data/80_20_test_results.csv", index=False)
    print(f"\nDetailed results saved to: src/data/80_20_test_results.csv")
    
    return results

if __name__ == "__main__":
    try:
        results = run_80_20_test()
        print("\n✅ 80-20 accuracy test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during 80-20 test: {str(e)}")
        import traceback
        traceback.print_exc() 
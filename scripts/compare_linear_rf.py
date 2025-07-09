import os
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends.csv"
MODELS_DIR = "county_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def compare_models():
    """
    Compare Linear Regression vs Random Forest performance
    """
    print("=== MODEL COMPARISON: LINEAR REGRESSION vs RANDOM FOREST ===\n")
    
    # Load data
    df = pd.read_csv(AGGREGATE_TRENDS_FILE)
    feature_cols = ["Population"] + [col for col in df.columns if col.startswith('monthly_average_')]
    X = df[feature_cols]
    y = df["SNAP_Applications"]
    
    # Drop rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print(f"Target variable: SNAP_Applications")
    print(f"Data range: {y.min():.0f} to {y.max():.0f} applications\n")
    
    # Split data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # === LINEAR REGRESSION ===
    print("=" * 60)
    print("LINEAR REGRESSION MODEL")
    print("=" * 60)
    
    # Train Linear Regression
    lr_model = LinearRegression().fit(X_train, y_train)
    
    # Predictions
    lr_train_pred = lr_model.predict(X_train)
    lr_test_pred = lr_model.predict(X_test)
    
    # Metrics
    lr_train_r2 = r2_score(y_train, lr_train_pred)
    lr_test_r2 = r2_score(y_test, lr_test_pred)
    lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
    lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
    lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
    lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
    
    print(f"Train R²: {lr_train_r2:.4f}")
    print(f"Test R²: {lr_test_r2:.4f}")
    print(f"Train MAE: {lr_train_mae:.2f}")
    print(f"Test MAE: {lr_test_mae:.2f}")
    print(f"Train RMSE: {lr_train_rmse:.2f}")
    print(f"Test RMSE: {lr_test_rmse:.2f}")
    
    # Cross-validation
    lr_cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R²: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std() * 2:.4f})")
    
    # Statistical significance (p-values)
    X_with_constant = sm.add_constant(X_train)
    lr_stats = sm.OLS(y_train, X_with_constant).fit()
    
    print("\nFeature Coefficients and P-values:")
    for feature, coef, pvalue in zip(['const'] + feature_cols, lr_stats.params, lr_stats.pvalues):
        significance = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""
        print(f"  {feature}: {coef:.6f} (p={pvalue:.6f}) {significance}")
    
    # === RANDOM FOREST ===
    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL")
    print("=" * 60)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    # Metrics
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
    rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    
    print(f"Train R²: {rf_train_r2:.4f}")
    print(f"Test R²: {rf_test_r2:.4f}")
    print(f"Train MAE: {rf_train_mae:.2f}")
    print(f"Test MAE: {rf_test_mae:.2f}")
    print(f"Train RMSE: {rf_train_rmse:.2f}")
    print(f"Test RMSE: {rf_test_rmse:.2f}")
    
    # Cross-validation
    rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R²: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = rf_model.feature_importances_
    print("\nFeature Importance:")
    for feature, importance in zip(feature_cols, feature_importance):
        print(f"  {feature}: {importance:.4f}")
    
    # === COMPARISON SUMMARY ===
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    comparison_data = {
        'Metric': ['Train R²', 'Test R²', 'Train MAE', 'Test MAE', 'Train RMSE', 'Test RMSE', 'CV R²'],
        'Linear Regression': [lr_train_r2, lr_test_r2, lr_train_mae, lr_test_mae, lr_train_rmse, lr_test_rmse, lr_cv_scores.mean()],
        'Random Forest': [rf_train_r2, rf_test_r2, rf_train_mae, rf_test_mae, rf_train_rmse, rf_test_rmse, rf_cv_scores.mean()]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Determine winner
    print(f"\n🏆 WINNER ANALYSIS:")
    if rf_test_r2 > lr_test_r2:
        print(f"  Random Forest wins on Test R²: {rf_test_r2:.4f} vs {lr_test_r2:.4f}")
    else:
        print(f"  Linear Regression wins on Test R²: {lr_test_r2:.4f} vs {rf_test_r2:.4f}")
    
    if rf_test_mae < lr_test_mae:
        print(f"  Random Forest wins on Test MAE: {rf_test_mae:.2f} vs {lr_test_mae:.2f}")
    else:
        print(f"  Linear Regression wins on Test MAE: {lr_test_mae:.2f} vs {rf_test_mae:.2f}")
    
    # Overfitting analysis
    lr_overfitting = lr_train_r2 - lr_test_r2
    rf_overfitting = rf_train_r2 - rf_test_r2
    
    print(f"\n📊 OVERFITTING ANALYSIS:")
    print(f"  Linear Regression overfitting: {lr_overfitting:.4f}")
    print(f"  Random Forest overfitting: {rf_overfitting:.4f}")
    
    if abs(lr_overfitting) < abs(rf_overfitting):
        print(f"  Linear Regression generalizes better")
    else:
        print(f"  Random Forest generalizes better")
    
    # Save both models
    with open(os.path.join(MODELS_DIR, "linear_regression_model.pkl"), "wb") as f:
        pickle.dump({"model": lr_model, "features": feature_cols, "type": "linear_regression"}, f)
    
    with open(os.path.join(MODELS_DIR, "random_forest_model.pkl"), "wb") as f:
        pickle.dump({"model": rf_model, "features": feature_cols, "type": "random_forest"}, f)
    
    print(f"\n✅ Models saved to {MODELS_DIR}/")
    print(f"  - linear_regression_model.pkl")
    print(f"  - random_forest_model.pkl")
    
    # Save comparison results
    comparison_df.to_csv("src/data/model_comparison_results.csv", index=False)
    print(f"✅ Comparison results saved to src/data/model_comparison_results.csv")
    
    return lr_model, rf_model, comparison_df

if __name__ == "__main__":
    compare_models() 
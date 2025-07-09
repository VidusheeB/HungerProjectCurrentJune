import os
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends.csv"

def analyze_negative_predictions():
    """
    Analyze which model generates more negative predictions and why
    """
    print("=== NEGATIVE PREDICTIONS ANALYSIS ===\n")
    
    # Load data
    df = pd.read_csv(AGGREGATE_TRENDS_FILE)
    feature_cols = ["Population"] + [col for col in df.columns if col.startswith('monthly_average_')]
    X = df[feature_cols]
    y = df["SNAP_Applications"]
    
    # Drop rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train both models
    lr_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
    
    # Get predictions
    lr_train_pred = lr_model.predict(X_train)
    lr_test_pred = lr_model.predict(X_test)
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    # === NEGATIVE PREDICTIONS ANALYSIS ===
    print("=" * 60)
    print("NEGATIVE PREDICTIONS COMPARISON")
    print("=" * 60)
    
    # Count negative predictions
    lr_train_neg = np.sum(lr_train_pred < 0)
    lr_test_neg = np.sum(lr_test_pred < 0)
    rf_train_neg = np.sum(rf_train_pred < 0)
    rf_test_neg = np.sum(rf_test_pred < 0)
    
    print(f"Linear Regression - Train negative predictions: {lr_train_neg}/{len(lr_train_pred)} ({lr_train_neg/len(lr_train_pred)*100:.2f}%)")
    print(f"Linear Regression - Test negative predictions: {lr_test_neg}/{len(lr_test_pred)} ({lr_test_neg/len(lr_test_pred)*100:.2f}%)")
    print(f"Random Forest - Train negative predictions: {rf_train_neg}/{len(rf_train_pred)} ({rf_train_neg/len(rf_train_pred)*100:.2f}%)")
    print(f"Random Forest - Test negative predictions: {rf_test_neg}/{len(rf_test_pred)} ({rf_test_neg/len(rf_test_pred)*100:.2f}%)")
    
    # Find actual negative predictions
    lr_neg_indices = np.where(lr_test_pred < 0)[0]
    rf_neg_indices = np.where(rf_test_pred < 0)[0]
    
    print(f"\nLinear Regression negative predictions in test set: {len(lr_neg_indices)}")
    print(f"Random Forest negative predictions in test set: {len(rf_neg_indices)}")
    
    # === WHY LINEAR REGRESSION GENERATES NEGATIVES ===
    print("\n" + "=" * 60)
    print("WHY LINEAR REGRESSION GENERATES NEGATIVE VALUES")
    print("=" * 60)
    
    # Linear regression equation: y = intercept + coef1*x1 + coef2*x2 + coef3*x3
    intercept = lr_model.intercept_
    coefficients = lr_model.coef_
    
    print(f"Linear Regression Equation:")
    print(f"  SNAP_Applications = {intercept:.2f} + {coefficients[0]:.6f}*Population + {coefficients[1]:.6f}*FoodBank + {coefficients[2]:.6f}*CalFresh")
    
    # Check when the equation becomes negative
    print(f"\nThe equation becomes negative when:")
    print(f"  {intercept:.2f} + {coefficients[0]:.6f}*Population + {coefficients[1]:.6f}*FoodBank + {coefficients[2]:.6f}*CalFresh < 0")
    
    # Analyze the negative predictions
    if len(lr_neg_indices) > 0:
        print(f"\nAnalyzing {len(lr_neg_indices)} negative Linear Regression predictions:")
        
        for i, idx in enumerate(lr_neg_indices[:5]):  # Show first 5
            actual = y_test.iloc[idx]
            predicted = lr_test_pred[idx]
            features = X_test.iloc[idx]
            
            print(f"\n  Sample {i+1}:")
            print(f"    Actual: {actual:.0f}")
            print(f"    Predicted: {predicted:.2f}")
            print(f"    Population: {features['Population']:.0f}")
            print(f"    FoodBank: {features['monthly_average_FoodBank']:.2f}")
            print(f"    CalFresh: {features['monthly_average_CalFresh']:.2f}")
            
            # Calculate each term
            pop_term = coefficients[0] * features['Population']
            foodbank_term = coefficients[1] * features['monthly_average_FoodBank']
            calfresh_term = coefficients[2] * features['monthly_average_CalFresh']
            total = intercept + pop_term + foodbank_term + calfresh_term
            
            print(f"    Equation breakdown:")
            print(f"      Intercept: {intercept:.2f}")
            print(f"      Population term: {pop_term:.2f}")
            print(f"      FoodBank term: {foodbank_term:.2f}")
            print(f"      CalFresh term: {calfresh_term:.2f}")
            print(f"      Total: {total:.2f}")
    
    # === WHY RANDOM FOREST GENERATES NEGATIVES ===
    print("\n" + "=" * 60)
    print("WHY RANDOM FOREST GENERATES NEGATIVE VALUES")
    print("=" * 60)
    
    print("Random Forest generates negative predictions when:")
    print("  1. The training data contains negative or very low SNAP application values")
    print("  2. The model extrapolates to regions outside the training data range")
    print("  3. The ensemble of trees predicts negative values")
    
    # Check training data range
    print(f"\nTraining data SNAP Applications range: {y_train.min():.0f} to {y_train.max():.0f}")
    print(f"Test data SNAP Applications range: {y_test.min():.0f} to {y_test.max():.0f}")
    
    # Analyze Random Forest negative predictions
    if len(rf_neg_indices) > 0:
        print(f"\nAnalyzing {len(rf_neg_indices)} negative Random Forest predictions:")
        
        for i, idx in enumerate(rf_neg_indices[:5]):  # Show first 5
            actual = y_test.iloc[idx]
            predicted = rf_test_pred[idx]
            features = X_test.iloc[idx]
            
            print(f"\n  Sample {i+1}:")
            print(f"    Actual: {actual:.0f}")
            print(f"    Predicted: {predicted:.2f}")
            print(f"    Population: {features['Population']:.0f}")
            print(f"    FoodBank: {features['monthly_average_FoodBank']:.2f}")
            print(f"    CalFresh: {features['monthly_average_CalFresh']:.2f}")
    
    # === COMPARISON SUMMARY ===
    print("\n" + "=" * 60)
    print("SUMMARY: WHICH MODEL GENERATES MORE NEGATIVES?")
    print("=" * 60)
    
    if lr_test_neg > rf_test_neg:
        print(f"🏆 Linear Regression generates MORE negative predictions")
        print(f"   Linear Regression: {lr_test_neg} negatives")
        print(f"   Random Forest: {rf_test_neg} negatives")
        print(f"\nREASON: Linear Regression can extrapolate beyond the data range")
        print(f"   - When feature combinations result in negative equation values")
        print(f"   - Especially with small populations and high FoodBank/CalFresh values")
    else:
        print(f"🏆 Random Forest generates MORE negative predictions")
        print(f"   Random Forest: {rf_test_neg} negatives")
        print(f"   Linear Regression: {lr_test_neg} negatives")
        print(f"\nREASON: Random Forest extrapolation behavior")
        print(f"   - When test data is outside training data range")
        print(f"   - Ensemble averaging can produce negative values")
    
    # === RECOMMENDATIONS ===
    print(f"\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("To handle negative predictions:")
    print("1. Set minimum prediction to 0 (SNAP applications can't be negative)")
    print("2. Use log transformation for training (if data allows)")
    print("3. Add constraints to Linear Regression")
    print("4. Use Random Forest with min_samples_leaf parameter")
    
    return lr_test_neg, rf_test_neg

if __name__ == "__main__":
    analyze_negative_predictions() 
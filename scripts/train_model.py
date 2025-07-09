import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import PoissonRegressor
import numpy as np

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends_scaled.csv"
MODELS_DIR = "county_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_global_model():
    df = pd.read_csv(AGGREGATE_TRENDS_FILE)
    # Use Population and all columns that start with 'monthly_average_' as features
    feature_cols = ["Population"] + [col for col in df.columns if col.startswith('monthly_average_')]
    X = df[feature_cols]
    y = df["SNAP_Applications"]
    # Drop rows with missing values in any feature or target
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Ensure target values are positive (SNAP applications cannot be negative)
    y = y.clip(lower=0)

    # Fit Random Forest model with constraints to prevent negative predictions
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Check for negative predictions on training data
    train_predictions = model.predict(X)
    negative_count = np.sum(train_predictions < 0)
    if negative_count > 0:
        print(f"⚠️  Warning: {negative_count} negative predictions in training data")
        print("   This will be handled by clipping predictions to 0 in production")
    
    r2 = model.score(X, y)
    print(f"Trained Random Forest model | R^2: {r2:.3f}")

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Feature importance
    feature_importance = model.feature_importances_
    print("\n=== FEATURE IMPORTANCE ===")
    for feature, importance in zip(feature_cols, feature_importance):
        print(f"{feature}: {importance:.4f}")

    # Model info
    print(f"\n=== MODEL INFO ===")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Max depth: {model.max_depth}")
    print(f"Min samples split: {model.min_samples_split}")
    print(f"Min samples leaf: {model.min_samples_leaf}")
    print(f"Model type: Random Forest (with post-processing to ensure non-negative predictions)")

    with open(os.path.join(MODELS_DIR, "global_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "features": feature_cols, "type": "random_forest"}, f)

if __name__ == "__main__":
    train_global_model()
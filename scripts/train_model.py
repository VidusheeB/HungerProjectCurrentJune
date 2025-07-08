import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

AGGREGATE_TRENDS_FILE = "src/data/aggregateTrends.csv"
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

    # Fit sklearn model for predictions
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    print(f"Trained global model | R^2: {r2:.3f}")

    # Fit statsmodels for p-values and statistical inference
    X_with_constant = sm.add_constant(X)
    model_stats = sm.OLS(y, X_with_constant).fit()
    
    print("\n=== STATSMODELS SUMMARY ===")
    print(model_stats.summary())
    
    print("\n=== P-VALUES ===")
    for feature, pvalue in zip(['const'] + feature_cols, model_stats.pvalues):
        print(f"{feature}: {pvalue:.6f}")

    with open(os.path.join(MODELS_DIR, "global_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "features": feature_cols}, f)

if __name__ == "__main__":
    train_global_model()
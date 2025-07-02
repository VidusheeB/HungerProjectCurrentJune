import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

AGGREGATE_DIR = "src/data/trends/aggregate"
SNAP_DATA_FILE = "src/data/SNAPApps/SNAPData.csv"  # Update if your SNAP data is elsewhere
MODELS_DIR = "county_models"

os.makedirs(MODELS_DIR, exist_ok=True)

def load_snap_data():
    # Assumes SNAP data has columns: county, month_year (e.g. 'May 2022'), SNAP_Applications, no header
    df = pd.read_csv(SNAP_DATA_FILE, header=None)
    df.columns = ["county", "month_year", "SNAP_Applications"]
    # Parse month-year like 'May 2022' to datetime (first of month)
    df["date"] = pd.to_datetime(df["month_year"], format="%b %Y", errors="coerce")
    # If some months are full names (e.g. 'March 2025'), try full month name too
    if df["date"].isna().any():
        df.loc[df["date"].isna(), "date"] = pd.to_datetime(df.loc[df["date"].isna(), "month_year"], format="%B %Y", errors="coerce")
    # Convert SNAP_Applications to numeric, '*' and empty to NaN
    df["SNAP_Applications"] = pd.to_numeric(df["SNAP_Applications"], errors="coerce")
    
    # Process each county separately
    processed_dfs = []
    for county, group in df.groupby('county'):
        # Sort by date
        group = group.sort_values('date')
        
        # Count non-null data points
        valid_data_points = group['SNAP_Applications'].notna().sum()
        
        # Only interpolate if we have some data but not enough (less than 6 months)
        if 0 < valid_data_points < 6:
            print(f"Interpolating data for {county} (only {valid_data_points} valid data points)")
            # Interpolate missing values using linear interpolation
            group['SNAP_Applications'] = group['SNAP_Applications'].interpolate(method='linear')
            # If there are still missing values at the beginning, forward fill
            group['SNAP_Applications'] = group['SNAP_Applications'].fillna(method='bfill')
            # If there are still missing values at the end, back fill
            group['SNAP_Applications'] = group['SNAP_Applications'].fillna(method='ffill')
        
        processed_dfs.append(group)
    
    # Combine back all counties
    df = pd.concat(processed_dfs)
    return df

def train_and_save_models():
    snap_df = load_snap_data()
    counties = [c for c in os.listdir(AGGREGATE_DIR) if os.path.isdir(os.path.join(AGGREGATE_DIR, c))]
    for county in counties:
        agg_path = os.path.join(AGGREGATE_DIR, county, "aggregate.csv")
        if not os.path.exists(agg_path):
            print(f"No aggregate data for {county}")
            continue
        trends_df = pd.read_csv(agg_path, parse_dates=["date"])
        # Get SNAP data for this county
        snap_county = snap_df[snap_df['county'] == county][['date', 'SNAP_Applications']]
        merged = pd.merge(trends_df, snap_county, on="date", how="inner")
        # Ensure correct SNAP_Applications column is used
        if "SNAP_Applications_y" in merged.columns:
            merged = merged.drop(columns=[col for col in merged.columns if col.startswith("SNAP_Applications_") and col != "SNAP_Applications_y"])
            merged = merged.rename(columns={"SNAP_Applications_y": "SNAP_Applications"})
        elif "SNAP_Applications_x" in merged.columns:
            merged = merged.rename(columns={"SNAP_Applications_x": "SNAP_Applications"})
        merged = merged.dropna(subset=["SNAP_Applications"])  # Only keep rows with SNAP data
        if merged.empty:
            print(f"No merged data for {county}")
            continue
        feature_cols = [col for col in merged.columns if col.startswith("trend_")]
        X = merged[feature_cols]
        y = merged["SNAP_Applications"]
        if X.empty or y.empty:
            print(f"Insufficient data for {county}")
            continue
        model = LinearRegression().fit(X, y)
        model_info = {
            "model": model,
            "features": feature_cols
        }
        with open(f"{MODELS_DIR}/{county}_model.pkl", "wb") as f:
            pickle.dump(model_info, f)
        print(f"Trained and saved model for {county}")

if __name__ == "__main__":
    train_and_save_models()
import os
import pandas as pd

TRENDS_BASE_DIR = "src/data/trends"
COUNTY_METRO_FILE = "src/data/county_to_metro.csv"
OUTPUT_BASE_DIR = os.path.join(TRENDS_BASE_DIR, "aggregate")

def get_keywords(trends_base_dir):
    return [name for name in os.listdir(trends_base_dir)
            if os.path.isdir(os.path.join(trends_base_dir, name))
            and name not in ["aggregate"]]

def normalize_metro_filename(metro):
    # Remove any trailing ' CA' or ' Ca' or ' ca'
    metro = metro.rstrip()
    if metro[-3:].lower() == ' ca':
        metro = metro[:-3]
    # Remove spaces and dashes, capitalize each word, join together
    return ''.join(word.capitalize() for word in metro.replace('-', ' ').split())

def load_trend_for_metro(metro, keyword):
    normalized_metro = normalize_metro_filename(metro)
    path = os.path.join(TRENDS_BASE_DIR, keyword, f"{normalized_metro}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        return None  # Not enough columns
    df.columns = ["date", f"trend_{keyword}"]
    
    # Handle weekly data format (YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    
    # Convert weekly data to monthly averages
    df["month"] = df["date"].dt.to_period('M')
    df = df.groupby("month").mean().reset_index()
    df["date"] = df["month"].dt.to_timestamp()
    df = df[["date", f"trend_{keyword}"]]
    
    return df

def load_snap_data_for_county(county):
    snap_path = os.path.join("src/data/SNAPApps", "SNAPData.csv")
    if not os.path.exists(snap_path):
        return None
    df = pd.read_csv(snap_path, header=None)
    df.columns = ["county", "month_year", "SNAP_Applications"]
    # Parse month-year format like 'May 2022'
    df["date"] = pd.to_datetime(df["month_year"], format="%b %Y", errors="coerce")
    county_mask = df["county"] == county
    county_df = df[county_mask].copy()
    county_df["SNAP_Applications"] = pd.to_numeric(county_df["SNAP_Applications"], errors="coerce")
    return county_df[["date", "SNAP_Applications"]].dropna(subset=["date"])
def aggregate_for_county(county, metro, keywords):
    dfs = []
    for keyword in keywords:
        df = load_trend_for_metro(metro, keyword)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    from functools import reduce
    agg = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), dfs)
    agg = agg.sort_values("date")
    # Merge SNAP data as last column
    snap_df = load_snap_data_for_county(county)
    if snap_df is not None and not snap_df.empty:
        agg = pd.merge(agg, snap_df, on='date', how='left')
    else:
        pass
    return agg

def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    county_metro = pd.read_csv(COUNTY_METRO_FILE)
    keywords = get_keywords(TRENDS_BASE_DIR)
    pass
    for _, row in county_metro.iterrows():
        county = row["county"]
        metro = row["metro_area"]
        pass
        agg = aggregate_for_county(county, metro, keywords)
        if agg is not None:
            county_dir = os.path.join(OUTPUT_BASE_DIR, county)
            os.makedirs(county_dir, exist_ok=True)
            agg.to_csv(os.path.join(county_dir, "aggregate.csv"), index=False)
            pass
        else:
            pass

if __name__ == "__main__":
    main()

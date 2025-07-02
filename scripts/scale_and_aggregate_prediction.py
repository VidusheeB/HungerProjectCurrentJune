import os
import pandas as pd
from datetime import datetime

# Configuration
PREDICTION_DATA_DIR = "src/data/prediction"
PREDICTION_AGGREGATE_DIR = "src/data/predictionAggregate"

# Ensure output directory exists
os.makedirs(PREDICTION_AGGREGATE_DIR, exist_ok=True)

def load_prediction_data(metro, keyword):
    """
    Load prediction data for a given metro area and keyword
    """
    prediction_file = os.path.join(PREDICTION_DATA_DIR, keyword, f"{metro}.csv")
    
    if not os.path.exists(prediction_file):
        print(f"No data for {metro} in {keyword}")
        return None
        
    try:
        # Read the CSV file
        df = pd.read_csv(prediction_file, header=None, names=['date', f'trend_{keyword}'])
        
        # Convert trend values to numeric, coercing errors to NaN
        trend_col = f'trend_{keyword}'
        df[trend_col] = pd.to_numeric(df[trend_col], errors='coerce')
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        
        # Drop rows with invalid dates or trend values
        df = df.dropna(subset=['date', trend_col])
        
        if df.empty:
            print(f"No valid data found in {prediction_file}")
            return None
            
        return df
        
    except Exception as e:
        print(f"Error processing {prediction_file}: {str(e)}")
        return None

def aggregate_prediction_data_for_metro(metro, keywords):
    """
    Aggregate prediction data for a metro area across all keywords
    """
    print(f"\nProcessing {metro}...")
    all_data = []
    
    # Load data for each keyword
    for keyword in keywords:
        df = load_prediction_data(metro, keyword)
        if df is not None:
            print(f"  - Loaded {keyword} data with {len(df)} rows")
            all_data.append(df)
    
    if not all_data:
        print(f"No prediction data found for {metro}")
        return None
    
    # Merge all dataframes on date
    merged = all_data[0]
    for df in all_data[1:]:
        merged = pd.merge(merged, df, on='date', how='outer')
    
    # Sort by date and forward fill missing values
    merged = merged.sort_values('date')
    
    if merged.empty:
        print("No valid data after merging")
        return None
    
    # Only keep the most recent entry for prediction
    latest = merged.iloc[-1:].copy()
    
    # Set date to next month for prediction
    latest['date'] = latest['date'] + pd.offsets.MonthBegin(1)
    
    print(f"  - Latest date: {latest['date'].iloc[0].strftime('%Y-%m-%d')}")
    print("  - Trend values:")
    for col in latest.columns:
        if col != 'date':
            # Safely format the value, handling both numeric and string types
            value = latest[col].iloc[0]
            if pd.api.types.is_numeric_dtype(latest[col]):
                print(f"    - {col}: {float(value):.2f}")
            else:
                print(f"    - {col}: {str(value)}")
    
    return latest

def main():
    print(f"Starting prediction aggregation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Looking for prediction data in: {PREDICTION_DATA_DIR}")
    
    # Create necessary directories
    os.makedirs(PREDICTION_AGGREGATE_DIR, exist_ok=True)
    
    # Get list of keywords (subdirectories in prediction data directory)
    try:
        keywords = [k for k in os.listdir(PREDICTION_DATA_DIR) 
                  if os.path.isdir(os.path.join(PREDICTION_DATA_DIR, k)) and 
                  not k.startswith('.')]  # Skip hidden directories
    except FileNotFoundError:
        print(f"Error: Prediction data directory not found: {PREDICTION_DATA_DIR}")
        return
    
    if not keywords:
        print(f"No keyword directories found in {PREDICTION_DATA_DIR}")
        return
    
    print(f"\nFound {len(keywords)} keyword directories: {', '.join(keywords)}")
    
    # Get list of metro areas (files in each keyword directory)
    metro_areas = set()
    for keyword in keywords:
        keyword_dir = os.path.join(PREDICTION_DATA_DIR, keyword)
        try:
            metros = [f.replace('.csv', '') for f in os.listdir(keyword_dir) 
                     if f.endswith('.csv') and not f.startswith('.')]
            metro_areas.update(metros)
            print(f"  - Found {len(metros)} metro areas in {keyword}")
        except FileNotFoundError:
            print(f"  - Warning: Keyword directory not found: {keyword_dir}")
            continue
    
    if not metro_areas:
        print("\nNo metro area files found. Please check your data directory structure.")
        print(f"Expected structure: {PREDICTION_DATA_DIR}/<keyword>/<metro>.csv")
        return
    
    print(f"\nFound {len(metro_areas)} unique metro areas to process")
    
    # Process each metro area
    successful = 0
    for metro in sorted(metro_areas):
        agg = aggregate_prediction_data_for_metro(metro, keywords)
        if agg is not None:
            metro_dir = os.path.join(PREDICTION_AGGREGATE_DIR, metro)
            os.makedirs(metro_dir, exist_ok=True)
            output_file = os.path.join(metro_dir, "aggregate.csv")
            
            try:
                agg.to_csv(output_file, index=False)
                print(f"  - Saved prediction data to {output_file}")
                successful += 1
            except Exception as e:
                print(f"  - Error saving {output_file}: {str(e)}")
    
    print(f"\nProcessing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully processed {successful} of {len(metro_areas)} metro areas")

if __name__ == "__main__":
    main()

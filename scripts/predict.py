import pandas as pd
import pickle
import sys
import os
import logging
from datetime import datetime, timedelta
from utils import scale_trends, normalize_trends_by_population  # <-- Import both scaling functions
import numpy as np # Added for np.nan

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "county_models"
COUNTY_METRO_MAP = "src/data/county_to_metro.csv"
POP_DATA_FILE = "src/data/popData.csv"
PREDICTION_BASE_DIR = "src/data/prediction"

# Always load the global model

def load_global_model():
    with open(os.path.join(MODELS_DIR, "global_model.pkl"), "rb") as f:
        return pickle.load(f)

def predict_next_month(trends_dict, population):
    if not trends_dict:
        raise ValueError("No trend data provided for prediction")
    
    model_info = load_global_model()
    features = model_info["features"]
    model = model_info["model"]
    # trends_dict: {"trend_keyword1": value, "trend_keyword2": value, ...}
    # Add population to the features
    input_dict = {**trends_dict, "Population": population}
    # Ensure all features are present
    X_pred = pd.DataFrame([input_dict], columns=features)
    
    # Check for missing features
    missing_features = []
    for feature in features:
        if pd.isna(X_pred[feature].iloc[0]):
            missing_features.append(feature)
    
    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")
    
    prediction = model.predict(X_pred)[0]
    
    # Ensure prediction is never negative (SNAP applications cannot be negative)
    prediction = max(0, prediction)
    
    return prediction

def get_metro_for_county(county):
    county_metro_map = pd.read_csv(COUNTY_METRO_MAP)
    row = county_metro_map[county_metro_map["county"] == county]
    if row.empty:
        raise ValueError(f"No metro found for county {county}")
    return row.iloc[0]["metro_area"]

def get_population_for_county(county):
    pop_df = pd.read_csv(POP_DATA_FILE)
    pop_df.columns = pop_df.columns.str.strip()
    row = pop_df[pop_df["County"] == county]
    if row.empty:
        raise ValueError(f"No population found for county {county}")
    return float(row.iloc[0]["Population"])

def get_latest_trends_for_metro(metro_area):
    """
    Load the latest trend data from the prediction folder for each keyword.
    Scale the prediction data to the training data using scale_trends, then normalize by population.
    """
    trends = {}
    if os.path.exists(PREDICTION_BASE_DIR):
        keywords = [d for d in os.listdir(PREDICTION_BASE_DIR)
                   if os.path.isdir(os.path.join(PREDICTION_BASE_DIR, d))]
    else:
        logger.error(f"Prediction base directory {PREDICTION_BASE_DIR} does not exist")
        return {}
    logger.info(f"Detected keywords: {keywords}")
    for keyword in keywords:
        prediction_file = os.path.join(PREDICTION_BASE_DIR, keyword, f"{metro_area}.csv")
        training_file = os.path.join("src/data/trends", keyword, f"{metro_area}.csv")
        if not os.path.exists(prediction_file):
            logger.warning(f"No prediction file found for {metro_area} in {keyword}")
            continue
        if not os.path.exists(training_file):
            logger.warning(f"No training file found for {metro_area} in {keyword}")
            continue
        # Load both training and prediction data
        df_train = pd.read_csv(training_file, header=None, names=['date', 'train_value'])
        
        # Handle Google Trends export format for prediction data
        try:
            # Read the file to find the actual data header
            with open(prediction_file, 'r') as f:
                lines = f.readlines()
            
            # Find the line that contains the actual data header (starts with Day,)
            header_line_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('Day,'):
                    header_line_idx = i
                    break
            
            if header_line_idx is None:
                logger.warning(f"No data header found in {prediction_file}")
                continue
            
            # Read the data starting from the header line
            df_pred = pd.read_csv(prediction_file, skiprows=header_line_idx)
            
            # The first column should be 'Day', rename it to 'date'
            # The second column should be the trend data, rename it to 'pred_value'
            if len(df_pred.columns) >= 2:
                df_pred.columns = ['date', 'pred_value'] + list(df_pred.columns[2:])
                df_pred = df_pred[['date', 'pred_value']]  # Keep only date and value columns
            else:
                logger.warning(f"Unexpected column structure in {prediction_file}")
                continue
                
        except Exception as e:
            logger.warning(f"Error reading prediction file {prediction_file}: {e}")
            continue
        
        # Remove Google Trends headers/metadata rows from prediction data
        known_headers = [
            'Category: All categories', 'Region:', 'Week', 'Day', 'Month', 'Year',
            'United States', 'State', 'City', 'Metro', 'Subregion', 'Search term',
            'Note:', 'Notes:', 'Interest over time', 'Interest by region', 'Top related queries', 'Rising related queries', 'Top', 'Rising', 'Keyword', 'Keywords', 'Time', 'Geo', 'isPartial', 'date', 'value', 'values', 'Average', 'Total', 'N/A', 'nan', '', None
        ]
        df_pred = df_pred[~df_pred['date'].astype(str).str.strip().isin(known_headers)]
        
        # Ensure both columns are numeric
        df_train['train_value'] = pd.to_numeric(df_train['train_value'], errors='coerce')
        df_pred['pred_value'] = pd.to_numeric(df_pred['pred_value'], errors='coerce')
        df_train = df_train[df_train['train_value'].notna()]
        df_pred = df_pred[df_pred['pred_value'].notna()]
        
        # Now parse dates
        df_train['date'] = pd.to_datetime(df_train['date'])
        df_pred['date'] = pd.to_datetime(df_pred['date'], errors='coerce')
        df_pred = df_pred[df_pred['date'].notna()]  # Only keep rows with valid dates
        
        # Aggregate daily prediction data to monthly averages
        if not df_pred.empty:
            df_pred['year_month'] = df_pred['date'].dt.to_period('M')
            monthly_pred = df_pred.groupby('year_month')['pred_value'].mean().reset_index()
            monthly_pred['date'] = monthly_pred['year_month'].dt.to_timestamp()
            monthly_pred = monthly_pred[['date', 'pred_value']]
            df_pred = monthly_pred
            logger.info(f"Processed {keyword} data for {metro_area}: {len(df_pred)} monthly records")
        else:
            logger.warning(f"No valid prediction data for {keyword} in {metro_area}")
            continue
        
        # Merge on date, keeping all dates
        df_merged = pd.merge(df_train, df_pred, on='date', how='outer')
        # Sort by date
        df_merged = df_merged.sort_values('date')
        # Scale prediction data to training data
        df_scaled = scale_trends(df_merged.copy(), 'train_value', 'pred_value')
        
        # Debug: Check if we have any valid prediction values after scaling
        valid_pred_values = df_scaled['pred_value'].dropna()
        if valid_pred_values.empty:
            logger.warning(f"No valid prediction values after scaling for {keyword} in {metro_area}")
            continue
        
        latest_pred_value = valid_pred_values.iloc[-1]
        logger.info(f"Latest {keyword} value for {metro_area}: {latest_pred_value}")
        
        # --- Population normalization ---
        # Need to know which county this metro_area maps to. Assume a function or mapping is available.
        # For now, try to infer county from metro_area (if only one county per metro, or use a mapping function)
        # This is a placeholder: you may need to adjust for your actual mapping logic.
        # We'll use the first county in the county-metro mapping for this metro.
        county_map_df = pd.read_csv('src/data/county_to_metro.csv')
        county_map_df.columns = county_map_df.columns.str.strip()
        counties_for_metro = county_map_df[county_map_df['metro_area'] == metro_area]['county'].tolist()
        if not counties_for_metro:
            logger.warning(f"No county found for metro area {metro_area}")
            continue
        county = counties_for_metro[0]
        # Create a DataFrame for normalization
        norm_df = pd.DataFrame({'county': [county], 'trend': [latest_pred_value]})
        norm_df = normalize_trends_by_population(norm_df, county_col='county', trend_cols=['trend'])
        
        # Debug: Check the normalized value
        normalized_value = norm_df['trend'].iloc[0]
        logger.info(f"Normalized {keyword} value for {county}: {normalized_value}")
        
        if pd.notna(normalized_value):
            trends[f"monthly_average_{keyword}"] = normalized_value
            logger.info(f"Successfully added monthly_average_{keyword} trend for {metro_area}: {normalized_value}")
        else:
            logger.warning(f"Normalized value is NaN for {keyword} in {metro_area}")
    return trends

def list_available_counties():
    """List all available counties that have population and metro mapping"""
    try:
        county_metro_map = pd.read_csv(COUNTY_METRO_MAP)
        pop_df = pd.read_csv(POP_DATA_FILE)
        counties = set(county_metro_map["county"]).intersection(set(pop_df["County"]))
        return sorted(counties)
    except Exception as e:
        logger.error(f"Error listing counties: {e}")
        return []

def zscore_to_flag(z):
    """Convert z-score to flag color."""
    if pd.isna(z):
        return 'Gray'
    if z < 0:
        return 'Green'
    if z >= 2:
        return 'Red'
    elif z >= 1:
        return 'Yellow'
    else:
        return 'Green'

def save_predictions_to_csv(predictions, output_file="src/data/finalPrediction.csv"):
    """Save predictions to a CSV file with risk flags."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'date': date,
                'county': county,
                'predicted_applications': pred
            }
            for (county, date), pred in predictions.items()
        ])
        
        # Load historical SNAP data to calculate county-specific statistics
        try:
            snap_data = pd.read_csv("src/data/SNAPApps/SNAPData.csv", header=None, 
                                   names=["county", "date_str", "SNAP_Applications"], thousands=",")
            snap_data["date"] = pd.to_datetime(snap_data["date_str"].str.strip(), format="%b %Y", errors="coerce")
            snap_data.loc[snap_data["date"].isna(), "date"] = pd.to_datetime(
                snap_data.loc[snap_data["date"].isna(), "date_str"].str.strip(), format="%B %Y", errors="coerce"
            )
            snap_data["SNAP_Applications"] = pd.to_numeric(snap_data["SNAP_Applications"].replace("*", pd.NA), errors="coerce")
            
            # Calculate county-specific statistics from historical data
            county_stats = snap_data.groupby('county')["SNAP_Applications"].agg(['mean', 'std']).reset_index()
            
            # Merge with predictions
            df = df.merge(county_stats, on='county', how='left')
            
            # Calculate z-scores using historical county-specific statistics
            df['z_score'] = (df['predicted_applications'] - df['mean']) / df['std']
            df['z_score'] = df['z_score'].fillna(0)  # Fill NaN with 0 if no historical data
            
        except Exception as e:
            print(f"Warning: Could not load historical data for county-specific z-scores: {str(e)}")
            print("Falling back to prediction-based z-scores...")
            
            # Fallback: Calculate z-scores for predictions (original method)
        mean_apps = df['predicted_applications'].mean()
        std_apps = df['predicted_applications'].std()
        
        # Handle case where all predictions are the same (std = 0)
        if std_apps == 0:
            df['z_score'] = 0
        else:
            df['z_score'] = (df['predicted_applications'] - mean_apps) / std_apps
        
        # Add flag based on z-score
        df['flag'] = df['z_score'].apply(zscore_to_flag)
        
        # Drop intermediate columns as they're not needed in the final output
        columns_to_drop = ['z_score']
        if 'mean' in df.columns:
            columns_to_drop.append('mean')
        if 'std' in df.columns:
            columns_to_drop.append('std')
        df = df.drop(columns=columns_to_drop)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Predictions with risk flags saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        return False

def generate_predictions(counties=None):
    """Generate predictions for all counties or specified counties."""
    if counties is None:
        counties = list_available_counties()
    
    predictions = {}
    today = datetime.now().date()
    # Generate prediction date (first day of current month)
    prediction_date = today.replace(day=1)
    prediction_date_str = prediction_date.strftime("%Y-%m-01")
    
    print(f"Generating predictions for {prediction_date_str}...")
    
    for county in counties:
        try:
            metro = get_metro_for_county(county)
            if not metro:
                print(f"Skipping {county}: No metro area found")
                continue
            
            trends = get_latest_trends_for_metro(metro)
            if not trends:
                print(f"Skipping {county}: No trend data available")
                continue
            
            population = get_population_for_county(county)
            prediction = predict_next_month(trends, population)
            predictions[(county, prediction_date_str)] = round(prediction, 2)
            print(f"{county}: {prediction:.2f}")
            
        except Exception as e:
            print(f"Error predicting for {county}: {str(e)}")
    
    return predictions

def main():
    if len(sys.argv) > 1:
        # Predict for specific county if provided
        county = sys.argv[1]
        try:
            metro = get_metro_for_county(county)
            if not metro:
                print(f"No metro area found for county: {county}")
                sys.exit(1)

            # Get latest trends for the metro area
            trends = get_latest_trends_for_metro(metro)
            if not trends:
                print(f"No trend data available for {metro}")
                sys.exit(1)

            population = get_population_for_county(county)
            # Make prediction
            prediction = predict_next_month(trends, population)
            print(f"Predicted SNAP applications for {county} next month: {prediction:.2f}")
            
            # Save to CSV
            save_predictions_to_csv({
                (county, datetime.now().strftime("%Y-%m-01")): round(prediction, 2)
            })

        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    else:
        # Generate predictions for all counties
        predictions = generate_predictions()
        if predictions:
            save_predictions_to_csv(predictions)

if __name__ == "__main__":
    main()
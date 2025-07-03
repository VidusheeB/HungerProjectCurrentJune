import pandas as pd
import pickle
import sys
import os
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "county_models"
COUNTY_METRO_MAP = "src/data/county_to_metro.csv"
PREDICTION_BASE_DIR = "src/data/prediction"

def load_model(county):
    with open(f"{MODELS_DIR}/{county}_model.pkl", "rb") as f:
        return pickle.load(f)

def predict_next_month(county, trends_dict):
    model_info = load_model(county)
    features = model_info["features"]
    model = model_info["model"]
    # trends_dict: {"trend_keyword1": value, "trend_keyword2": value, ...}
    X_pred = pd.DataFrame([trends_dict], columns=features)
    prediction = model.predict(X_pred)[0]
    return prediction

def get_metro_for_county(county):
    county_metro_map = pd.read_csv(COUNTY_METRO_MAP)
    row = county_metro_map[county_metro_map["county"] == county]
    if row.empty:
        raise ValueError(f"No metro found for county {county}")
    return row.iloc[0]["metro_area"]

def get_latest_trends_for_metro(metro_area):
    """
    Load the latest trend data from the prediction folder for each keyword
    """
    trends = {}
    keywords = ["CalFresh", "ElectronicBenefitTransfer", "SNAP"]
    
    for keyword in keywords:
        prediction_file = os.path.join(PREDICTION_BASE_DIR, keyword, f"{metro_area}.csv")
        if not os.path.exists(prediction_file):
            logger.warning(f"No prediction file found for {metro_area} in {keyword}")
            continue
        
        try:
            # Skip the first two lines (header and category)
            df = pd.read_csv(prediction_file, skiprows=2, header=None, names=["date", "value"])
            
            # Convert value to numeric, coercing errors to NaN
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Convert date and handle any parsing errors
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
            
            # Drop rows with invalid dates or values
            df = df.dropna(subset=["date", "value"])
            
            if df.empty:
                logger.warning(f"No valid data in {prediction_file}")
                continue
                
            # Filter out zeros and get the latest value
            non_zero = df[df["value"] > 0]
            if non_zero.empty:
                logger.warning(f"No non-zero values in {prediction_file}")
                continue
                
            latest = non_zero.sort_values("date").iloc[-1]
            trends[f"trend_{keyword}"] = float(latest["value"])
            
            logger.info(f"Using {keyword} trend value: {latest['value']:.2f}")
            
        except Exception as e:
            logger.warning(f"Error processing {prediction_file}: {str(e)}")
            continue
    
    if not trends:
        raise ValueError(f"No valid trend data found for metro area {metro_area}")
    return trends

def list_available_counties():
    """List all available counties that have models"""
    try:
        counties = [f.replace('_model.pkl', '') for f in os.listdir(MODELS_DIR) 
                   if f.endswith('_model.pkl')]
        return sorted(counties)
    except FileNotFoundError:
        logger.error(f"Models directory not found: {MODELS_DIR}")
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
                
            prediction = predict_next_month(county, trends)
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

            # Make prediction
            prediction = predict_next_month(county, trends)
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
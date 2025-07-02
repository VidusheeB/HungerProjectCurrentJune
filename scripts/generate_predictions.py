import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODELS_DIR = "county_models"
COUNTY_METRO_MAP = "src/data/county_to_metro.csv"
PREDICTION_BASE_DIR = "src/data/prediction"
OUTPUT_FILE = "src/data/finalPrediction.csv"

def load_model(county):
    """Load a trained model for a specific county."""
    try:
        with open(f"{MODELS_DIR}/{county}_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model for {county}: {str(e)}")
        return None

def get_metro_for_county(county):
    """Get the metro area for a given county."""
    try:
        county_metro_map = pd.read_csv(COUNTY_METRO_MAP)
        row = county_metro_map[county_metro_map["county"] == county]
        if row.empty:
            logger.warning(f"No metro found for county {county}")
            return None
        return row.iloc[0]["metro_area"]
    except Exception as e:
        logger.error(f"Error getting metro for {county}: {str(e)}")
        return None

def get_latest_trends(metro_area):
    """Get the latest trend data for a metro area."""
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
            if not df.empty:
                # Get the most recent non-null value
                latest = df.dropna().iloc[-1]["value"]
                trends[f"trend_{keyword}"] = latest
        except Exception as e:
            logger.error(f"Error processing {prediction_file}: {str(e)}")
    
    return trends if trends else None

def predict_for_county(county):
    """Generate prediction for a single county."""
    metro = get_metro_for_county(county)
    if not metro:
        return None
    
    trends = get_latest_trends(metro)
    if not trends:
        logger.warning(f"No trend data available for {county} (metro: {metro})")
        return None
    
    model_info = load_model(county)
    if not model_info:
        return None
    
    try:
        features = model_info["features"]
        model = model_info["model"]
        
        # Prepare features in the correct order
        X_pred = pd.DataFrame([{f: trends.get(f, 0) for f in features}], columns=features)
        prediction = model.predict(X_pred)[0]
        
        return prediction
    except Exception as e:
        logger.error(f"Error predicting for {county}: {str(e)}")
        return None

def generate_predictions():
    """Generate predictions for all counties and save to CSV."""
    logger.info("Starting prediction generation...")
    
    # Get list of all counties with models
    try:
        counties = [f.replace("_model.pkl", "") for f in os.listdir(MODELS_DIR) 
                   if f.endswith("_model.pkl")]
    except Exception as e:
        logger.error(f"Error listing model files: {str(e)}")
        return
    
    predictions = []
    today = datetime.now().date()
    
    # Generate prediction date (first day of next month)
    if today.day > 1:
        next_month = today.replace(day=1) + timedelta(days=32)
        prediction_date = next_month.replace(day=1)
    else:
        prediction_date = today.replace(day=1)
    
    prediction_date_str = prediction_date.strftime("%Y-%m-01")
    
    logger.info(f"Generating predictions for {prediction_date_str}...")
    
    # Generate predictions for each county
    for county in counties:
        prediction = predict_for_county(county)
        if prediction is not None:
            predictions.append({
                'date': prediction_date_str,
                'county': county,
                'predicted_applications': round(prediction, 2)
            })
    
    # Save to CSV
    if predictions:
        df = pd.DataFrame(predictions)
        df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Predictions saved to {OUTPUT_FILE}")
        logger.info(f"Generated {len(predictions)} predictions")
    else:
        logger.warning("No predictions were generated")

if __name__ == "__main__":
    generate_predictions()

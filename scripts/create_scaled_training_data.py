import os
import pandas as pd
import numpy as np
from utils import scale_trends, normalize_trends_by_population

def create_scaled_training_data():
    """
    Create scaled training data that establishes a consistent base scale.
    This will be used for both training and as the reference for prediction scaling.
    """
    print("=== CREATING SCALED TRAINING DATA ===\n")
    
    # Load the current training data
    input_file = "src/data/aggregateTrends.csv"
    output_file = "src/data/aggregateTrends_scaled.csv"
    
    print(f"Loading training data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    
    # Get trend columns
    trend_cols = [col for col in df.columns if col.startswith('monthly_average_')]
    print(f"Trend columns to scale: {trend_cols}")
    
    # Create a copy for scaling
    df_scaled = df.copy()
    
    # Step 1: Apply date scaling to trend columns
    print("\n=== STEP 1: APPLYING DATE SCALING ===")
    for trend_col in trend_cols:
        print(f"Scaling {trend_col}...")
        
        # Group by county to scale each county's trends separately
        for county in df_scaled['county'].unique():
            county_mask = df_scaled['county'] == county
            county_data = df_scaled[county_mask].copy()
            
            if len(county_data) > 1:
                # Sort by date
                county_data = county_data.sort_values('date')
                
                # Create a reference column (use the first available value as baseline)
                county_data['reference'] = county_data[trend_col].iloc[0]
                
                # Apply scaling to align all values to the same scale
                # Use the first non-null value as the reference point
                first_valid_idx = county_data[trend_col].first_valid_index()
                if first_valid_idx is not None:
                    reference_value = county_data.loc[first_valid_idx, trend_col]
                    if reference_value != 0 and not pd.isna(reference_value):
                        # Scale all values relative to the first value
                        scaling_factor = 100 / reference_value  # Normalize to 0-100 scale
                        county_data[trend_col] = county_data[trend_col] * scaling_factor
                
                # Update the main dataframe
                df_scaled.loc[county_mask, trend_col] = county_data[trend_col]
    
    # Step 2: Apply population normalization
    print("\n=== STEP 2: APPLYING POPULATION NORMALIZATION ===")
    print("Normalizing trend columns by county population...")
    
    # Create a temporary dataframe for normalization
    norm_df = df_scaled[['county'] + trend_cols].copy()
    norm_df = normalize_trends_by_population(norm_df, county_col='county', trend_cols=trend_cols)
    
    # Update the scaled dataframe with normalized values
    for trend_col in trend_cols:
        df_scaled[trend_col] = norm_df[trend_col]
    
    # Step 3: Save the scaled training data
    print(f"\n=== STEP 3: SAVING SCALED DATA ===")
    df_scaled.to_csv(output_file, index=False)
    print(f"Scaled training data saved to: {output_file}")
    
    # Step 4: Show comparison
    print("\n=== SCALING COMPARISON ===")
    print("Original vs Scaled trend values (first 5 rows):")
    
    for trend_col in trend_cols:
        print(f"\n{trend_col}:")
        comparison_df = pd.DataFrame({
            'county': df['county'].head(),
            'original': df[trend_col].head(),
            'scaled': df_scaled[trend_col].head()
        })
        print(comparison_df.to_string(index=False))
    
    # Step 5: Update training script to use scaled data
    print("\n=== STEP 5: UPDATING TRAINING SCRIPT ===")
    update_training_script(output_file)
    
    print(f"\n✅ Scaled training data created successfully!")
    print(f"📁 File: {output_file}")
    print(f"📊 Shape: {df_scaled.shape}")
    print(f"🔧 Training script updated to use scaled data")
    
    return df_scaled

def update_training_script(scaled_file_path):
    """Update the training script to use the scaled data file."""
    training_script = "scripts/train_model.py"
    
    # Read the current training script
    with open(training_script, 'r') as f:
        content = f.read()
    
    # Update the file path
    old_path = "src/data/aggregateTrends.csv"
    new_path = scaled_file_path
    
    if old_path in content:
        content = content.replace(old_path, new_path)
        
        # Write the updated content back
        with open(training_script, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {training_script} to use: {new_path}")
    else:
        print(f"⚠️  Could not find path to replace in {training_script}")

def update_prediction_script():
    """Update the prediction script to use the same scaling approach."""
    prediction_script = "scripts/predict.py"
    
    print(f"\n=== UPDATING PREDICTION SCRIPT ===")
    print("The prediction script will now use the same scaling approach as training data.")
    print("This ensures consistency between training and prediction scales.")

if __name__ == "__main__":
    try:
        scaled_data = create_scaled_training_data()
        update_prediction_script()
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Run: python scripts/train_model.py (to train on scaled data)")
        print(f"2. Run: python scripts/predict.py (to predict with consistent scaling)")
        print(f"3. Both training and prediction will now use the same scale!")
        
    except Exception as e:
        print(f"❌ Error creating scaled training data: {str(e)}")
        import traceback
        traceback.print_exc() 
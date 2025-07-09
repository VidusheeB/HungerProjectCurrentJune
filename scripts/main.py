import sys
import traceback

# Step 1: Add metro area to popData
print("\n=== STEP 1: Add metro area to popData.csv ===")
try:
    from add_metro_to_popdata import add_metro_area_to_popdata
    add_metro_area_to_popdata()
except Exception as e:
    print(f"Error in add_metro_to_popdata: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Create aggregate trends
print("\n=== STEP 2: Create aggregate trends ===")
try:
    from create_aggregate_trends import create_aggregate_trends
    create_aggregate_trends()
except Exception as e:
    print(f"Error in create_aggregate_trends: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Interpolate missing SNAP data
print("\n=== STEP 3: Interpolate missing SNAP data ===")
try:
    from interpolate_missing_snap_data import interpolate_missing_snap_data
    interpolate_missing_snap_data()
except Exception as e:
    print(f"Error in interpolate_missing_snap_data: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Scale training data
print("\n=== STEP 4: Scale training data ===")
try:
    from create_scaled_training_data import create_scaled_training_data
    create_scaled_training_data()
except Exception as e:
    print(f"Error in create_scaled_training_data: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 5: Train model
print("\n=== STEP 5: Train global model ===")
try:
    from train_model import train_global_model
    train_global_model()
except Exception as e:
    print(f"Error in train_model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 6: Run predictions
print("\n=== STEP 6: Run predictions ===")
try:
    from predict import main as predict_main
    predict_main()
except Exception as e:
    print(f"Error in predict: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 7: Aggregate predictions
print("\n=== STEP 7: Aggregate predictions ===")
try:
    from scale_and_aggregate_prediction import main as agg_pred_main
    agg_pred_main()
except Exception as e:
    print(f"Error in scale_and_aggregate_prediction: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Pipeline complete! ===")

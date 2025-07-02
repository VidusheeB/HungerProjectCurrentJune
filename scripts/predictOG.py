import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from datetime import datetime, date
import os
import sys

def detect_anomalies(data, window_size=6):
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()
    z_scores = (data - rolling_mean) / rolling_std
    anomalies = (z_scores.abs() > 2)
    return anomalies, z_scores

def predict_snap_applications():
    try:
        project_root = Path(__file__).parent.parent
        dashboard_path = project_root / '2019-2023_dashboard.csv'
        if not dashboard_path.exists():
            print(f"Error: Dashboard file not found at {dashboard_path}")
            sys.exit(1)
        df = pd.read_csv(dashboard_path)
        df = df.sort_values(['State', 'County', 'Year', 'Month'])
        predictions = []
        # Prepare San Diego trends data
        sdc_trend_path = project_root / 'src' / 'data' / 'SanDiego.csv'
        if sdc_trend_path.exists():
            sdc_trend = pd.read_csv(sdc_trend_path)
            sdc_trend['CalFresh'] = pd.to_numeric(sdc_trend['CalFresh'], errors='coerce')
        else:
            sdc_trend = None
        for county, county_data in df.groupby(['State', 'County', 'fipsValue']):
            state = county[0]
            county_name = county[1]
            fips = county[2]
            snap_data = county_data['SNAP_Applications'].values
            years = county_data['Year'].values
            months = county_data['Month'].values

            # Detect anomalies in SNAP applications
            anomalies, z_scores = detect_anomalies(pd.Series(snap_data))
            flags = []
            for z in z_scores:
                if pd.isnull(z):
                    flags.append('Gray')
                elif z < 0:
                    flags.append('Green')
                elif z >= 2:
                    flags.append('Red')
                elif z >= 1:
                    flags.append('Yellow')
                else:
                    flags.append('Green')
            latest_flag = flags[-1] if len(flags) > 0 else 'Gray'
            latest_z = z_scores.iloc[-1] if len(z_scores) > 0 else None

            # Prepare regression features
            X = np.arange(len(snap_data)).reshape(-1, 1)
            y = snap_data

            # Special logic for San Diego
            if county_name.strip().lower().replace(' county','') == 'san diego' and sdc_trend is not None:
                # Merge on year/month
                merged = county_data.copy()
                merged = merged.merge(
                    sdc_trend,
                    left_on=['Year', 'Month'],
                    right_on=['year', 'month'],
                    how='left'
                )
                search_cols = [
                    'Supplemental Nutrition Assistance Program',
                    'Food bank',
                    'Soup kitchen',
                    'CalFresh',
                    'Electronic Benefit Transfer'
                ]
                # Use all search columns as regression features
                search_features = []
                for col in search_cols:
                    if col in merged.columns:
                        search_features.append(merged[col].astype(float).fillna(0))
                    else:
                        search_features.append(pd.Series([0]*len(merged)))
                X_feat = np.column_stack([np.arange(len(snap_data))] + search_features)
                model = LinearRegression()
                model.fit(X_feat, y)
                # Prepare next month's features
                next_features = []
                for col in search_cols:
                    if col in merged.columns and not merged[col].empty:
                        next_features.append(merged[col].iloc[-1])
                    else:
                        next_features.append(0)
                next_X = np.array([[len(snap_data)] + next_features])
                prediction = model.predict(next_X)[0]
                extra_fields = {
                    'Trend_Spike_Columns': '',  # Optionally keep info about spikes if you wish
                }
            else:
                # Standard regression for other counties
                model = LinearRegression()
                model.fit(X, y)
                next_X = np.array([[len(snap_data)]])
                prediction = model.predict(next_X)[0]
                extra_fields = {}

            # Output fields
            latest_year = years[-1]
            latest_month = months[-1]
            # Compute next month/year
            if latest_month == 12:
                pred_month = 1
                pred_year = latest_year + 1
            else:
                pred_month = latest_month + 1
                pred_year = latest_year
            pred_month_label = f"{pred_year}-{str(pred_month).zfill(2)}"

            # Add population and density if available
            population = county_data['Population'].iloc[-1] if 'Population' in county_data.columns else ''
            pdensity = county_data['PDensity'].iloc[-1] if 'PDensity' in county_data.columns else ''

            pred_row = {
                'State': state,
                'County': county_name,
                'fipsValue': fips,
                'Prediction_Month': pred_month_label,
                'Predicted': round(prediction, 2),
                'Flag': latest_flag,
                'Latest_Z_Score': latest_z,
                'Population': population,
                'PDensity': pdensity
            }
            pred_row.update(extra_fields)
            predictions.append(pred_row)

        # Save predictions to CSV
        pred_df = pd.DataFrame(predictions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / 'src' / 'data'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"snap_predictions_{timestamp}.csv"
        pred_df.to_csv(output_path, index=False)
        # Also save a copy with a generic name for easy access
        pred_df.to_csv(output_dir / "snap_predictions_latest.csv", index=False)
        print(f"Predictions saved to {output_path}")
        return predictions

    except Exception as e:
        print("Error in predict_snap_applications:", e)
        raise

if __name__ == '__main__':
    predict_snap_applications()
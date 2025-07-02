# from pathlib import Path
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from datetime import datetime

# def detect_anomalies(data, window_size=6):
#     rolling_mean = data.rolling(window=window_size).mean()
#     rolling_std = data.rolling(window=window_size).std()
#     z_scores = (data - rolling_mean) / rolling_std
#     anomalies = (z_scores.abs() > 2)
#     return anomalies, z_scores

# def predict_snap_applications_all_counties():
#     try:
#         project_root = Path.cwd()
#         dashboard_path = project_root / '2019-2023_dashboard.csv'
#         trends_dir = project_root / 'src' / 'data' / 'trends'

#         if not dashboard_path.exists():
#             return "Error: Dashboard file not found."

#         df = pd.read_csv(dashboard_path)
#         df = df.sort_values(['State', 'County', 'Year', 'Month'])
#         predictions = []

#         for county, county_data in df.groupby(['State', 'County', 'fipsValue']):
#             state = county[0]
#             county_name = county[1]
#             fips = county[2]
#             snap_data = county_data['SNAP_Applications'].values
#             years = county_data['Year'].values
#             months = county_data['Month'].values

#             anomalies, z_scores = detect_anomalies(pd.Series(snap_data))
#             flags = []
#             for z in z_scores:
#                 if pd.isnull(z):
#                     flags.append('Gray')
#                 elif z < 0:
#                     flags.append('Green')
#                 elif z >= 2:
#                     flags.append('Red')
#                 elif z >= 1:
#                     flags.append('Yellow')
#                 else:
#                     flags.append('Green')
#             latest_flag = flags[-1] if len(flags) > 0 else 'Gray'
#             latest_z = z_scores.iloc[-1] if len(z_scores) > 0 else None

#             X = np.arange(len(snap_data)).reshape(-1, 1)
#             y = snap_data

#             county_filename = county_name.strip().replace(" ", "").replace("County", "") + ".csv"
#             trend_path = trends_dir / county_filename

#             if trend_path.exists():
#                 trend_df = pd.read_csv(trend_path)
#                 search_cols = [
#                     'Supplemental Nutrition Assistance Program',
#                     'Food bank',
#                     'Soup kitchen',
#                     'CalFresh',
#                     'Electronic Benefit Transfer'
#                 ]
#                 for col in search_cols:
#                     if col in trend_df.columns:
#                         trend_df[col] = pd.to_numeric(trend_df[col], errors='coerce')

#                 merged = county_data.copy()
#                 merged = merged.merge(
#                     trend_df,
#                     left_on=['Year', 'Month'],
#                     right_on=['year', 'month'],
#                     how='left'
#                 )

#                 search_features = []
#                 for col in search_cols:
#                     if col in merged.columns:
#                         shifted = merged[col].astype(float).fillna(0).shift(1)
#                         search_features.append(shifted)
#                     else:
#                         search_features.append(pd.Series([0] * len(merged)))

#                 X_feat = np.column_stack([np.arange(len(snap_data))] + search_features)
#                 model = LinearRegression()
#                 model.fit(X_feat[1:], y[1:])  # skip first due to shift

#                 next_features = []
#                 for col in search_cols:
#                     if col in merged.columns and not merged[col].empty:
#                         next_features.append(merged[col].iloc[-1])
#                     else:
#                         next_features.append(0)
#                 next_X = np.array([[len(snap_data)] + next_features])
#                 prediction = model.predict(next_X)[0]
#             else:
#                 model = LinearRegression()
#                 model.fit(X, y)
#                 next_X = np.array([[len(snap_data)]])
#                 prediction = model.predict(next_X)[0]

#             latest_year = years[-1]
#             latest_month = months[-1]
#             if latest_month == 12:
#                 pred_month = 1
#                 pred_year = latest_year + 1
#             else:
#                 pred_month = latest_month + 1
#                 pred_year = latest_year
#             pred_month_label = f"{pred_year}-{str(pred_month).zfill(2)}"

#             population = county_data['Population'].iloc[-1] if 'Population' in county_data.columns else ''
#             pdensity = county_data['PDensity'].iloc[-1] if 'PDensity' in county_data.columns else ''

#             pred_row = {
#                 'State': state,
#                 'County': county_name,
#                 'fipsValue': fips,
#                 'Prediction_Month': pred_month_label,
#                 'Predicted': round(prediction, 2),
#                 'Flag': latest_flag,
#                 'Latest_Z_Score': latest_z,
#                 'Population': population,
#                 'PDensity': pdensity
#             }
#             predictions.append(pred_row)

#         pred_df = pd.DataFrame(predictions)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = project_root / 'src' / 'data'
#         output_dir.mkdir(parents=True, exist_ok=True)
#         output_path = output_dir / f"snap_predictions_{timestamp}.csv"
#         pred_df.to_csv(output_path, index=False)
#         pred_df.to_csv(output_dir / "snap_predictions_latest.csv", index=False)
#         return f"Predictions saved to {output_path}"

#     except Exception as e:
#         return f"Error in predict_snap_applications: {e}"

# predict_snap_applications_all_counties()

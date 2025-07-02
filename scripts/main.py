# import os
# import sys
# import json
# import argparse
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from pathlib import Path

# pd.set_option('future.no_silent_downcasting', True)

# #############
# # Utility Functions
# #############

# def snake_case(string):
#     return string.strip().replace(" ", "_")

# def most_current_file(current_dir_region):
#     max_date = None
#     max_file_path = None

#     for file_path in current_dir_region.glob('*.csv'):
#         file_date_str = file_path.stem
#         try:
#             file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
#         except ValueError:
#             continue
#         if max_date is None or file_date > max_date:
#             max_date = file_date
#             max_file_path = file_path
#     return max_file_path

# def make_date(date_string, ft='%Y-%m-%d'):
#     return datetime.strptime(date_string, ft)

# #############
# # Configurations
# #############

# # Define the main directory where your CSVs are stored
# main_directory = Path.cwd() / "data"  # uses ./data as root

# # Subdirectories for outputs
# scaled_dir = Path.cwd() / "scaled"
# monthly_dir = Path.cwd() / "monthly"

# # Make sure the folders exist
# scaled_dir.mkdir(parents=True, exist_ok=True)
# monthly_dir.mkdir(parents=True, exist_ok=True)

# #############
# # Processing Functions
# #############

# def scale_trends(df, previous_column, current_column):
#     scaling_factor = 1
#     max_value_previous_year = df[previous_column].max()
#     date_of_max_value_previous_year = df.loc[df[previous_column].idxmax(), 'date']
#     max_value_current_year = df[current_column].max()
#     date_of_max_value_current_year = df.loc[df[current_column].idxmax(), 'date']

#     if max_value_previous_year != 0 and date_of_max_value_previous_year < date_of_max_value_current_year:
#         scaled_value = df.loc[df['date'] == date_of_max_value_previous_year, previous_column].values[0]
#         current_value = df.loc[df['date'] == date_of_max_value_previous_year, current_column].values[0]
#         if np.isnan(current_value):
#             first_date = df.loc[(df[current_column] != 0) & (df[current_column].notna()), 'date'].iloc[0]
#             first_value = df.loc[df['date'] == first_date, current_column].values[0]
#             previous_value = df.loc[df['date'] == first_date, previous_column].values[0]
#             scaling_factor = previous_value / first_value
#         else:
#             scaling_factor = scaled_value / current_value

#     df[current_column] = df[current_column] * scaling_factor
#     df[current_column] = df[previous_column].combine_first(df[current_column])
#     return df

# def process_csv_files():
#     for region_dir in main_directory.glob('*'):
#         if region_dir.is_dir():
#             region_name = region_dir.name
#             csv_files = sorted(region_dir.glob('*.csv'))

#             if not csv_files:
#                 continue

#             merged_df = pd.DataFrame()
#             for csv_file in csv_files:
#                 df = pd.read_csv(csv_file)
#                 df['date'] = pd.to_datetime(df['date'])
#                 col_name = csv_file.stem
#                 df.rename(columns={df.columns[1]: col_name}, inplace=True)
#                 if merged_df.empty:
#                     merged_df = df
#                 else:
#                     merged_df = pd.merge(merged_df, df, on='date', how='outer')

#             merged_df.sort_values('date', inplace=True)
#             merged_df.reset_index(drop=True, inplace=True)

#             # Scale the data
#             df_columns = merged_df.columns[1:].tolist()
#             df_columns.sort()
#             for idx in range(len(df_columns)-1):
#                 previous_col = df_columns[idx]
#                 current_col = df_columns[idx+1]
#                 merged_df = scale_trends(merged_df, previous_col, current_col)

#             # Save scaled data
#             scaled_output = scaled_dir / f"{snake_case(region_name)}.csv"
#             merged_df.to_csv(scaled_output, index=False)
#             print(f"Saved scaled data for {region_name} to {scaled_output}")

# #############
# # Main Entrypoint
# #############export PATH="$HOME/Libexport PATH="$HOME/Library/Python/3.9/bin:$PATH"rary/Python/3.9/bin:$PATH"export PATH="$HOME/Library/Python/3.9/bin:$PATH"

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--process", action='store_true', help="Process existing CSV data")
#     args = parser.parse_args()

#     if args.process:
#         process_csv_files()
#         print("Data processing complete.")
#     else:
#         print("No action specified. Use --process to process data.")

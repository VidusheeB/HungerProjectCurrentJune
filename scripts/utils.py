import numpy as np
import pandas as pd

def scale_trends(df, previous_column, current_column):
    """
    Scale the current period's trends data to match the scale of the previous period's data.
    This is important for Google Trends data, which is relative per period.
    """
    scaling_factor = 1
    max_value_previous_year = df[previous_column].max()
    date_of_max_value_previous_year = df.loc[df[previous_column].idxmax(), 'date']
    max_value_current_year = df[current_column].max()
    date_of_max_value_current_year = df.loc[df[current_column].idxmax(), 'date']

    if max_value_previous_year != 0 and date_of_max_value_previous_year < date_of_max_value_current_year:
        scaled_value = df.loc[df['date'] == date_of_max_value_previous_year, previous_column].values[0]
        current_value = df.loc[df['date'] == date_of_max_value_previous_year, current_column].values[0]
        if np.isnan(current_value):
            first_date = df.loc[(df[current_column] != 0) & (df[current_column].notna()), 'date'].iloc[0]
            first_value = df.loc[df['date'] == first_date, current_column].values[0]
            previous_value = df.loc[df['date'] == first_date, previous_column].values[0]
            scaling_factor = previous_value / first_value
        else:
            scaling_factor = scaled_value / current_value

    df[current_column] = df[current_column] * scaling_factor
    df[current_column] = df[previous_column].combine_first(df[current_column])
    return df


def normalize_trends_by_population(df, county_col='county', trend_cols=None, popdata_path="src/data/popData.csv"):
    """
    Normalize trend columns by county population.
    Args:
        df: DataFrame with a 'county' column and trend columns.
        county_col: Name of the county column.
        trend_cols: List of trend columns to normalize. If None, all columns except county_col are used.
        popdata_path: Path to the population data CSV.
    Returns:
        DataFrame with trend columns divided by population.
    """
    pop_df = pd.read_csv(popdata_path)
    pop_map = pop_df.set_index('County')['Population'].to_dict()
    if trend_cols is None:
        trend_cols = [col for col in df.columns if col != county_col]
    df = df.copy()
    df['__pop'] = df[county_col].map(pop_map)
    for col in trend_cols:
        df[col] = df[col] / df['__pop']
    df = df.drop(columns='__pop')
    return df 
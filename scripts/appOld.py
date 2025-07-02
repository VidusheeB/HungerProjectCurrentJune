import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from datetime import datetime

# --- CONFIGURATION ---
TITLE = "California Food Assistance Dashboard"
SUBTITLE = "SNAP food assistance application collected by google trends."
GEOJSON_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson"

# --- LOAD DATA ---
@st.cache_data
def load_snap_data():
    try:
        # Read the CSV with proper column names
        df = pd.read_csv(
            "src/data/SNAPApps/SNAPData.csv",
            header=None,
            names=["county", "date_str", "SNAP_Applications"],
            thousands=","
        )
        
        # Convert date string to datetime (handling both full and abbreviated month names)
        df["date"] = pd.to_datetime(df["date_str"].str.strip(), format="%b %Y", errors="coerce")
        # For any dates that didn't parse (might be full month names), try the full month format
        if df["date"].isna().any():
            df.loc[df["date"].isna(), "date"] = pd.to_datetime(
                df.loc[df["date"].isna(), "date_str"].str.strip(), 
                format="%B %Y",
                errors="coerce"
            )
        
        # Convert SNAP_Applications to numeric, handling non-numeric values
        df["SNAP_Applications"] = pd.to_numeric(df["SNAP_Applications"], errors="coerce")
        
        # Extract year and month for filtering
        df["Year"] = df["date"].dt.year
        df["Month"] = df["date"].dt.month
        
        return df
        
    except FileNotFoundError:
        st.error("SNAP data file not found. Please ensure the file exists at src/data/SNAPApps/SNAPData.csv")
        st.stop()
    except Exception as e:
        st.error(f"Error loading SNAP data: {str(e)}")
        st.stop()

@st.cache_data
def load_geojson():
    response = requests.get(GEOJSON_URL)
    response.raise_for_status()
    return response.json()

# --- UI LAYOUT ---
st.set_page_config(page_title=TITLE, layout="wide")

# --- MENU BAR ---
tabs = st.tabs(["Current Map", "Predictions Map", "Data Table"])

with tabs[0]:
    st.title(TITLE)
    st.markdown(f"<h4 style='margin-top:-12px;color:gray'>{SUBTITLE}</h4>", unsafe_allow_html=True)

    # Load SNAP data and GeoJSON
    snap_df = load_snap_data()
    counties_geojson = load_geojson()
    
    if snap_df is None:
        st.error("Failed to load SNAP data.")
        st.stop()
    
    # Debug: Print unique counties in the data
    if 'county' in snap_df.columns:
        st.sidebar.write(f"Unique counties in SNAP data: {len(snap_df['county'].unique())}")
    if 'date' in snap_df.columns:
        st.sidebar.write(f"Date range in SNAP data: {snap_df['date'].min()} to {snap_df['date'].max()}")

    # Clean and normalize county names for consistent matching
    def clean_county_name(name):
        # Remove 'County' and any extra whitespace, then lowercase
        return str(name).replace(' County', '').strip().lower()
    
    # Clean county names in the main dataframe
    snap_df['county_clean'] = snap_df['county'].apply(clean_county_name)
    
    # Prepare all_counties for use in all scopes
    all_counties = [feature['properties']['name'] for feature in counties_geojson['features']]
    
    # Create a mapping from clean names to original GeoJSON names
    geojson_name_map = {
        clean_county_name(feature['properties']['name']): feature['properties']['name']
        for feature in counties_geojson['features']
    }
    
    # Update county names in the dataframe to match GeoJSON exactly
    snap_df['county'] = snap_df['county_clean'].map(geojson_name_map).fillna(snap_df['county'])
    
    # Create normalized county names for matching (use clean names for consistency)
    snap_df['county_normalized'] = snap_df['county_clean']

    # --- MONTH/YEAR SELECTION ---
    unique_dates = snap_df['date'].sort_values().unique()
    date_options = [d.strftime('%b %Y') for d in unique_dates]
    selected_date = st.selectbox("Select Month", options=date_options, index=len(date_options)-1)
    selected_date_dt = pd.to_datetime(selected_date)

    # Filter for the selected date
    filtered_df = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].copy()
    
    # Debug: Check for missing data
    if not filtered_df.empty:
        # Print all counties in the data for the selected date
        st.sidebar.write(f"### Raw data for {selected_date}:")
        st.sidebar.write(filtered_df[['county', 'SNAP_Applications']].sort_values('county'))
        
        # Check for missing values
        missing_data = filtered_df[filtered_df['SNAP_Applications'].isna()]
        if not missing_data.empty:
            st.sidebar.warning(f"Warning: Found {len(missing_data)} counties with missing SNAP data for {selected_date}")
            st.sidebar.write("Counties with missing data:")
            st.sidebar.dataframe(missing_data[['county', 'SNAP_Applications']])
        else:
            st.sidebar.success("No missing SNAP data found for the selected date")

    # --- COLOR MAPPING BASED ON Z-SCORE OF PREVIOUS DATA ---
    flag_color_map = {
        "Red": "#e74c3c",      # High risk
        "Yellow": "#f7ca18",   # Elevated risk
        "Green": "#27ae60",    # Stable risk
        "Gray": "#888888",     # No data (darker gray)
        None: "#888888"
    }
    risk_map = {'Red': 'High', 'Yellow': 'Elevated', 'Green': 'Stable', 'Gray': 'Insufficient Data', None: 'Insufficient Data'}

    def compute_z_scores(df):
        # Calculate mean and std across all training data for each county
        county_stats = df.groupby('county_normalized')['SNAP_Applications'].agg(['mean', 'std'])
        
        # Merge stats back to the main dataframe
        df = df.merge(county_stats, on='county_normalized', how='left', suffixes=('', '_county'))
        
        # Calculate z-scores using the training data statistics
        z_scores = (df['SNAP_Applications'] - df['mean']) / df['std']
        
        # Handle cases where std is 0 (constant values) or NaN (single data point)
        z_scores = z_scores.fillna(0)  # If std is 0, z-score is 0 (no deviation)
        
        return z_scores

    # Compute z-scores using training data statistics
    snap_df['z_score'] = compute_z_scores(snap_df)

    # Get z-scores for the selected date
    zscore_map = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].set_index('county_normalized')['z_score'].to_dict()

    def zscore_to_flag(z):
        if pd.isnull(z):
            return 'Gray'
        if z < 0:
            return 'Green'
        if z >= 2:
            return 'Red'
        elif z >= 1:
            return 'Yellow'
        else:
            return 'Green'

    # Prepare filtered_df for map
    filtered_df['county_normalized'] = filtered_df['county'].str.strip().str.lower()
    filtered_df['z_score'] = filtered_df['county_normalized'].map(zscore_map)
    filtered_df['Flag'] = filtered_df['z_score'].apply(zscore_to_flag)
    filtered_df['color'] = filtered_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))

    # Debug: Print all counties in GeoJSON for comparison
    geojson_counties = [feature['properties']['name'].strip().lower() 
                       for feature in counties_geojson['features']]
    st.sidebar.write("### GeoJSON Counties (first 5):", sorted(geojson_counties)[:5])
    
    # Ensure every county in GeoJSON is present in filtered_df for the selected month
    counties_in_map = set()
    counties_in_data = set(filtered_df['county_normalized'].unique())
    st.sidebar.write("### Counties in data (first 5):", sorted(counties_in_data)[:5])
    
    for county in counties_geojson['features']:
        county_name = county['properties']['name']
        county_norm = clean_county_name(county_name)
        counties_in_map.add(county_norm)
        
        if county_norm not in counties_in_data:
            # Debug: Check if this is a known county with data but mismatched name
            matching_data = filtered_df[filtered_df['county_normalized'].str.contains(county_norm, case=False, na=False)]
            if not matching_data.empty:
                st.sidebar.warning(f"Potential name mismatch for {county_name} - found similar: {matching_data['county'].unique()}")
            
            # Add county with default values
            new_row = {
                'county': county_name,
                'Region': 'No data available',
                'Population': 'No data available',
                'PDensity': 'No data available',
                'SNAP_Applications': None,  # Explicitly set to None
                'Predicted': 'No data available',
                'Flag': 'Gray',
                'date': selected_date_dt,
                'color': flag_color_map['Gray'],
                'county_normalized': county_norm,
                'z_score': None
            }
            filtered_df = pd.concat([filtered_df, pd.DataFrame([new_row])], ignore_index=True)

    def normalize_county(name):
        return str(name).strip().lower().replace(' county', '')

    # Format SNAP_Applications as integer with thousands separator if it's a number
    def format_snap(val):
        try:
            if pd.isnull(val):
                return 'No data available'
            v = float(val)
            if v.is_integer():
                return f"{int(v):,}"
            return f"{v:,.0f}"
        except Exception:
            return str(val)
    filtered_df['SNAP_Applications'] = filtered_df['SNAP_Applications'].apply(format_snap)
    filtered_df['color'] = filtered_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))

    # --- MAP PLOTTING ---
    fig = px.choropleth(
        filtered_df,
        geojson=counties_geojson,
        locations='county',  # Use county name for matching
        color='Flag',
        color_discrete_map=flag_color_map,
        hover_name=None,
        custom_data=['county', 'SNAP_Applications'],
        scope="usa",
        labels={},
        featureidkey="properties.name",
        height=700
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>SNAP Applications: %{customdata[1]:,}<extra></extra>"
    )
    fig.update_geos(
        fitbounds="locations",  # Zoom to California, matching Predictions Map
        visible=True,
        showcountries=True,
        showsubunits=True,
        scope="usa"
    )
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        legend_title_text="Risk Level",
        showlegend=False  # Hide Plotly's built-in legend
    )

    # --- CUSTOM LEGEND/KEY (TOP RIGHT) ---
    st.markdown("""
    <div style='position:absolute;right:20px;top:180px;background:white;padding:12px 18px 12px 12px;border-radius:8px;border:1px solid #eee;z-index:1000;max-width:220px;'>
    <b>Key</b><br>
    <span style='color:#e74c3c'>&#9632;</span> High risk<br>
    <span style='color:#f7ca18'>&#9632;</span> Elevated risk<br>
    <span style='color:#27ae60'>&#9632;</span> Stable risk<br>
    <span style='color:#bdc3c7'>&#9632;</span> No data available<br>
    </div>
    """, unsafe_allow_html=True)

    # --- IN-PLACE MAP FOR ANIMATION ---
    map_placeholder = st.empty()
    map_placeholder.plotly_chart(fig, use_container_width=True)

    # Debug: Show county matching stats
    st.sidebar.write("## County Data Matching")
    st.sidebar.write(f"Counties in map: {len(counties_in_map)}")
    st.sidebar.write(f"Counties with data: {len(counties_in_data)}")
    matched = counties_in_map.intersection(counties_in_data)
    st.sidebar.write(f"Counties matched: {len(matched)} ({len(matched)/len(counties_in_map)*100:.1f}%)")
    
    # Show sample of counties not found in data
    missing = counties_in_map - counties_in_data
    if missing:
        st.sidebar.write("\n**Sample of counties missing data:**")
        for county in sorted(list(missing))[:5]:
            st.sidebar.write(f"- {county}")
    
    # Check for data quality issues
    missing_data = filtered_df[filtered_df['SNAP_Applications'].isna()]
    if not missing_data.empty:
        st.sidebar.warning(f"\n**Warning:** {len(missing_data)} counties have missing SNAP data")
        st.sidebar.write("Sample of affected counties:")
        for _, row in missing_data.sort_values('county').head(3).iterrows():
            st.sidebar.write(f"- {row['county']} (ID: {row['county_normalized']})")
    
    # Debug: Show sample of counties with data
    has_data = filtered_df[~filtered_df['SNAP_Applications'].isna()]
    if not has_data.empty:
        st.sidebar.write("\n**Sample of counties with data:**")
        for _, row in has_data.sort_values('county').head(3).iterrows():
            st.sidebar.write(f"- {row['county']}: {row['SNAP_Applications']}")

with tabs[1]:
    # --- TITLE ---
    st.title("Predictions")
    # --- CUSTOM LEGEND/KEY (TOP RIGHT, ABSOLUTE POSITION) ---
    st.markdown("""
    <div style='position:absolute;right:20px;top:180px;background:white;padding:12px 18px 12px 12px;border-radius:8px;border:1px solid #eee;z-index:1000;max-width:220px;'>
    <b>Key</b><br>
    <span style='color:#e74c3c'>&#9632;</span> High risk<br>
    <span style='color:#f7ca18'>&#9632;</span> Elevated risk<br>
    <span style='color:#27ae60'>&#9632;</span> Stable risk<br>
    <span style='color:#bdc3c7'>&#9632;</span> No data available<br>
    </div>
    """, unsafe_allow_html=True)
    map_placeholder = st.empty()
    # --- LOAD DATA ---
    snap_df = load_snap_data()
    counties_geojson = load_geojson()
    snap_df['county'] = snap_df['county'].str.replace(' County', '', regex=False)
    unique_dates = snap_df['date'].sort_values().unique()
    date_options = [d.strftime('%b %Y') for d in unique_dates]

    # --- Add predicted month option from predictions CSV ---
    import glob, os
    pred_csv_files = glob.glob(os.path.join('src', 'data', 'snap_predictions_*.csv'))
    def normalize_county(name):
        return str(name).strip().lower().replace(' county', '')
    pred_month_label = None
    if pred_csv_files:
        latest_pred_csv = max(pred_csv_files, key=os.path.getctime)
        pred_df = pd.read_csv(latest_pred_csv)
        if 'county' in pred_df.columns:
            pred_df['county'] = pred_df['county'].astype(str)
            pred_df['county_normalized'] = pred_df['county'].str.strip().str.lower()
        else:
            st.warning("Prediction data missing 'county' column")
        if 'Prediction_Month' in pred_df.columns and not pred_df.empty:
            # Compute the true predicted month as the month after the latest in dashboard data
            latest_actual_date = snap_df['date'].max()
            if latest_actual_date.month == 12:
                pred_year = latest_actual_date.year + 1
                pred_month = 1
            else:
                pred_year = latest_actual_date.year
                pred_month = latest_actual_date.month + 1
            pred_month_label = f"{pred_year}-{str(pred_month).zfill(2)}"
            pred_dt = pd.to_datetime(pred_month_label, format='%Y-%m')
            pred_month_english = pred_dt.strftime('%b %Y')
            pred_option = f"Predicted: {pred_month_english}"
            date_options_with_pred = date_options + [pred_option]
        else:
            date_options_with_pred = date_options
    else:
        date_options_with_pred = date_options
        pred_df = None
        pred_flag_map = {}

    selected_date = st.selectbox("Select Month", options=date_options_with_pred, index=len(date_options_with_pred)-1, key="pred_month")
    # Determine if user selected predicted month
    use_predicted = pred_month_label and selected_date == f"Predicted: {pred_month_english}"
    if use_predicted and pred_df is not None:
        # Use only the predictions for the computed predicted month
        pred_month_rows = pred_df[pred_df['Prediction_Month'] == pred_month_label]
        pred_flag_map = pred_month_rows.set_index('county_normalized')['Flag'].to_dict()
        # Use latest available data for base, but color by prediction
        filtered_df = snap_df[snap_df['date'] == unique_dates[-1]].copy()
        filtered_df['county_normalized'] = filtered_df['county'].str.strip().str.lower()
        filtered_df['Flag'] = filtered_df['county_normalized'].map(pred_flag_map).fillna('Gray')
        filtered_df['color'] = filtered_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))
    else:
        # Use actuals for map coloring (z-score risk logic, like Current Map tab)
        if selected_date in date_options:
            selected_date_dt = pd.to_datetime(selected_date)
            filtered_df = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].copy()
            filtered_df['county_normalized'] = filtered_df['county'].str.strip().str.lower()
            # Compute z-scores using training data statistics for predictions tab
            snap_df['county_normalized'] = snap_df['county'].str.strip().str.lower()
            snap_df = snap_df.sort_values(['county_normalized', 'date'])
            snap_df['z_score'] = compute_z_scores(snap_df[snap_df['date'] <= selected_date_dt])
            zscore_map = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].set_index('county_normalized')['z_score'].to_dict()
            def zscore_to_flag(z):
                if pd.isnull(z):
                    return 'Gray'
                if z < 0:
                    return 'Green'
                if z >= 2:
                    return 'Red'
                elif z >= 1:
                    return 'Yellow'
                else:
                    return 'Green'
            filtered_df['z_score'] = filtered_df['county_normalized'].map(zscore_map)
            filtered_df['Flag'] = filtered_df['z_score'].apply(zscore_to_flag)
            filtered_df['color'] = filtered_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))
        else:
            filtered_df = snap_df[snap_df['date'] == unique_dates[-1]].copy()
            filtered_df['Flag'] = 'Gray'
            filtered_df['color'] = flag_color_map['Gray']


    # Ensure every county in GeoJSON is present in filtered_df for the selected month
    all_counties = [feature['properties']['name'] for feature in counties_geojson['features']]
    filtered_df = filtered_df.set_index('county')
    for county in all_counties:
        if county not in filtered_df.index:
            filtered_df.loc[county] = {
                'Region': 'No data available',
                'Population': 'No data available',
                'PDensity': 'No data available',
                'SNAP_Applications': 'No data available',
                'Predicted': 'No data available',
                'Flag': 'Gray',
                'date': selected_date_dt,
                'color': flag_color_map['Gray'],
                'county_normalized': county.strip().lower(),
                'z_score': None
            }
    filtered_df = filtered_df.reset_index()

    def normalize_county(name):
        return str(name).strip().lower().replace(' county', '')

    # Initialize prediction variables
    pred_map = {}
    # Use current month as the prediction month
    from datetime import datetime
    current_date = datetime.now()
    pred_month_english = current_date.strftime('%b %Y')
    
    # Load county to metro mapping
    county_metro_path = os.path.join('src', 'data', 'county_to_metro.csv')
    if os.path.exists(county_metro_path):
        county_metro = pd.read_csv(county_metro_path)
        county_metro['county_normalized'] = county_metro['county'].apply(normalize_county)
        county_metro_map = county_metro.set_index('county_normalized')['metro_area'].to_dict()
    else:
        county_metro_map = {}
    
    # Get the latest prediction data for each metro
    pred_base_dir = os.path.join('src', 'data', 'prediction', 'SNAP')
    if os.path.exists(pred_base_dir):
        # Get current month's predictions
        current_month = current_date.strftime('%Y-%m')
        
        # For each metro area's prediction file
        for metro_file in os.listdir(pred_base_dir):
            if metro_file.endswith('.csv'):
                metro_name = os.path.splitext(metro_file)[0]
                metro_path = os.path.join(pred_base_dir, metro_file)
                
                try:
                    # Read the prediction file, skipping the first line (category)
                    df = pd.read_csv(metro_path, skiprows=1)
                    if df.empty or len(df.columns) < 2:
                        continue
                        
                    # Rename columns and parse dates
                    df.columns = ['date', 'prediction']
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Get the latest prediction for the current month
                    current_month_preds = df[df['date'].dt.strftime('%Y-%m') == current_month]
                    if not current_month_preds.empty:
                        # Use the most recent prediction for the month
                        latest_pred = current_month_preds.iloc[-1]['prediction']
                        
                        # Find all counties in this metro area and update their predictions
                        counties = [c for c, m in county_metro_map.items() if m == metro_name]
                        for county in counties:
                            pred_map[county] = latest_pred
                            
                except Exception as e:
                    print(f"Error processing {metro_path}: {e}")
    
    def get_prediction_hover(row):
        county_norm = normalize_county(row['county'])
        pred_val = pred_map.get(county_norm, 'No data available')
        if pd.isnull(pred_val) or pred_val == 'No data available':
            return f"Prediction for {pred_month_english}: No data available"
        try:
            pred_val = float(pred_val)
            pred_val_str = f"{int(pred_val):,}" if pred_val.is_integer() else f"{pred_val:,.0f}"
        except Exception:
            pred_val_str = str(pred_val)
        return f"Prediction for {pred_month_english}: {pred_val_str}"
    filtered_df['prediction_str'] = filtered_df.apply(get_prediction_hover, axis=1)
    fig = px.choropleth(
        filtered_df,
        geojson=counties_geojson,
        locations='county',
        color='Flag',
        color_discrete_map=flag_color_map,
        hover_name=None,
        custom_data=['county', 'prediction_str'],
        scope="usa",
        labels={"Flag": "Z-Score Risk Level"},
        featureidkey="properties.name",
        height=700
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
    )
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.title("Data Table")
    import glob, os
    pred_csv_files = glob.glob(os.path.join('src', 'data', 'snap_predictions_*.csv'))
    if pred_csv_files:
        latest_pred_csv = max(pred_csv_files, key=os.path.getctime)
        pred_df = pd.read_csv(latest_pred_csv)
        pred_df['County'] = pred_df['County'].astype(str)
        # Load dashboard data
        dash_df = load_snap_data()
        dash_df['County'] = dash_df['County'].astype(str)
        # Month/year dropdown
        dash_df['date'] = pd.to_datetime(dash_df['Year'].astype(str) + '-' + dash_df['Month'].astype(str).str.zfill(2) + '-01')
        all_dates = dash_df['date'].sort_values().unique()
        date_options = [d.strftime('%b %Y') for d in all_dates]
        selected_date = st.selectbox("Select Month", options=date_options, index=len(date_options)-1, key="data_table_month")
        selected_date_dt = pd.to_datetime(selected_date)
        # Filter dashboard data for selected month
        month_df = dash_df[dash_df['date'].dt.strftime('%b %Y') == selected_date].copy()
        # Merge predictions with dashboard data (left join on County)
        merged = pd.merge(pred_df, month_df, on='County', how='left', suffixes=('_pred', '_actual'))
        # Use the Flag from the predictions CSV for risk level
        st.write('Unique Flags in pred_df:', pred_df['Flag'].unique())
        st.write('Unique Flags in merged:', merged['Flag'].unique())
        merged['Risk Level'] = merged['Flag'].map(risk_map).fillna('Insufficient Data')
        st.write('Unique Risk Levels in merged:', merged['Risk Level'].unique())
        pred_col = 'Predicted_pred' if 'Predicted_pred' in merged.columns else ('Predicted' if 'Predicted' in merged.columns else None)
        # Ensure pred_month_label is defined
        if 'Prediction_Month' in pred_df.columns and not pred_df.empty:
            pred_month_label = pred_df['Prediction_Month'].iloc[0]
        else:
            pred_month_label = 'Unknown'
        # Fill missing static county info from dashboard for each county using dash_df
        for col in ['Region', 'Population', 'PDensity']:
            if col in merged.columns:
                missing = merged[col].isnull()
                merged.loc[missing, col] = merged.loc[missing, 'County'].map(
                    dash_df.drop_duplicates('County').set_index('County')[col]
                )
        # Restore full table layout: County, Region, Population, PDensity, SNAP Applications, Prediction, Risk Level
        columns_to_show = [
            'County', 'Region', 'Population', 'PDensity', 'SNAP_Applications', pred_col, 'Risk Level'
        ]
        columns_present = [col for col in columns_to_show if col in merged.columns or col == 'Risk Level']
        table = merged[columns_present].copy()
        table.rename(columns={pred_col: f'Predicted: {pred_month_label}'}, inplace=True)
        # Format Population as integer string, all other columns as string for consistent left alignment
        if 'Population' in table.columns:
            table['Population'] = table['Population'].apply(lambda x: str(int(float(x))) if pd.notnull(x) and x != '' else '')
        for col in table.columns:
            if col != 'Population':
                table[col] = table[col].apply(lambda x: str(x) if pd.notnull(x) else "")
        # Style rows
        def highlight_row(row):
            if row['Risk Level'] == 'High':
                return ['background-color: #ffe5e5']*len(row)
            elif row['Risk Level'] == 'Elevated':
                return ['background-color: #fffbe5']*len(row)
            else:
                return ['']*len(row)
        st.dataframe(table, use_container_width=True)
    else:
        st.warning("No predictions file found.")

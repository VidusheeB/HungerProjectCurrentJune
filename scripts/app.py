import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import glob, os
from datetime import datetime

# --- CONFIGURATION ---
TITLE = "California Food Assistance Dashboard"
SUBTITLE = "SNAP food assistance application collected by Google Trends."
GEOJSON_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson"

# --- LOAD SNAP DATA ---
@st.cache_data
def load_snap_data():
    df = pd.read_csv(
        "src/data/SNAPApps/SNAPData.csv",
        header=None,
        names=["county", "date_str", "SNAP_Applications"],
        thousands=","  # in case of thousands separator
    )
    df["date"] = pd.to_datetime(df["date_str"].str.strip(), format="%b %Y", errors="coerce")
    df.loc[df["date"].isna(), "date"] = pd.to_datetime(
        df.loc[df["date"].isna(), "date_str"].str.strip(), format="%B %Y", errors="coerce"
    )
    df["SNAP_Applications"] = pd.to_numeric(df["SNAP_Applications"].replace("*", pd.NA), errors="coerce")
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    return df

# --- LOAD GEOJSON ---
@st.cache_data
def load_geojson():
    response = requests.get(GEOJSON_URL)
    response.raise_for_status()
    geojson = response.json()
    for feature in geojson['features']:
        name = feature['properties'].get('name', '')
        feature['properties']['name'] = name.replace(' County', '').strip()
    return geojson

# --- UTILITIES ---
def format_snap(val):
    try:
        if pd.isnull(val):
            return 'No data available'
        v = float(val)
        return f"{int(v):,}" if v.is_integer() else f"{v:,.0f}"
    except Exception:
        return str(val)

def zscore_to_flag(z):
    if pd.isnull(z): return 'Gray'
    if z < 0: return 'Green'
    if z >= 2: return 'Red'
    elif z >= 1: return 'Yellow'
    return 'Green'

def compute_z_scores(df):
    county_stats = df.groupby('county')["SNAP_Applications"].agg(['mean', 'std'])
    df = df.merge(county_stats, on='county', how='left')
    z_scores = (df["SNAP_Applications"] - df["mean"]) / df["std"]
    return z_scores.fillna(0)

# --- LAYOUT ---
st.set_page_config(page_title=TITLE, layout="wide")
tabs = st.tabs(["Current Map", "Predictions Map"])

# --- TAB 1: CURRENT MAP ---
with tabs[0]:
    st.title(TITLE)
    st.markdown(f"<h4 style='margin-top:-12px;color:gray'>{SUBTITLE}</h4>", unsafe_allow_html=True)

    snap_df = load_snap_data()
    counties_geojson = load_geojson()

    unique_dates = snap_df['date'].sort_values().unique()
    date_options = [d.strftime('%b %Y') for d in unique_dates]
    selected_date = st.selectbox("Select Month", options=date_options, index=len(date_options)-1)

    filtered_df = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].copy()
    filtered_df['SNAP_Applications_Display'] = filtered_df['SNAP_Applications'].apply(format_snap)

    snap_df['z_score'] = compute_z_scores(snap_df)
    zscore_map = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].set_index('county')['z_score'].to_dict()
    filtered_df['z_score'] = filtered_df['county'].map(zscore_map)
    filtered_df['Flag'] = filtered_df['z_score'].apply(zscore_to_flag)

    flag_color_map = {
        "Red": "#e74c3c",
        "Yellow": "#f7ca18",
        "Green": "#27ae60",
        "Gray": "#888888",
        None: "#888888"
    }
    filtered_df['color'] = filtered_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))

    fig = px.choropleth(
        filtered_df,
        geojson=counties_geojson,
        locations='county',
        color='Flag',
        color_discrete_map=flag_color_map,
        custom_data=['county', 'SNAP_Applications_Display'],
        featureidkey="properties.name",
        scope="usa",
        height=700
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>SNAP Applications: %{customdata[1]}<extra></extra>"
    )
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PREDICTIONS MAP ---
# Define flag color map at the top level
flag_color_map = {
    "Red": "#e74c3c",
    "Yellow": "#f7ca18",
    "Green": "#27ae60",
    "Gray": "#888888",
    None: "#888888"
}

with tabs[1]:
    st.title("Predictions")
    snap_df = load_snap_data()
    counties_geojson = load_geojson()
    snap_df['county'] = snap_df['county'].astype(str)
    unique_dates = snap_df['date'].sort_values().unique()
    date_options = [d.strftime('%b %Y') for d in unique_dates]

    pred_path = os.path.join('src', 'data', 'finalPrediction.csv')
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        pred_df['county'] = pred_df['county'].astype(str)
        pred_df['county_normalized'] = pred_df['county'].str.strip().str.lower()

        # Set predicted month to current month and year
        from datetime import datetime
        current_date = datetime.now()
        pred_month_label = current_date.strftime("%Y-%m")
        pred_dt = pd.to_datetime(pred_month_label, format='%Y-%m')
        pred_month_english = pred_dt.strftime('%b %Y')
        pred_option = f"Predicted: {pred_month_english}"
        date_options_with_pred = date_options + [pred_option]
    else:
        pred_df = None
        date_options_with_pred = date_options

    selected_date = st.selectbox("Select Month", options=date_options_with_pred, index=len(date_options_with_pred)-1, key="pred_month")
    use_predicted = pred_df is not None and selected_date == f"Predicted: {pred_month_english}"

    if use_predicted:
        pred_rows = pred_df[pred_df['date'] == pred_month_label].copy()
        pred_map = pred_rows.set_index('county_normalized')['predicted_applications'].to_dict()
        
        # Ensure flag column exists and is properly formatted
        if 'flag' not in pred_rows.columns:
            pred_rows['flag'] = 'Gray'  # Default to Gray if flag column is missing
        
        # Debug: Print unique flag values
        print("Unique flag values in pred_rows:", pred_rows['flag'].unique())
        
        pred_flag_map = pred_rows.set_index('county_normalized')['flag'].str.capitalize().to_dict()
        print("Sample flag mappings:", list(pred_flag_map.items())[:5])

        base_df = snap_df[snap_df['date'] == unique_dates[-1]].copy()
        base_df['county_normalized'] = base_df['county'].str.strip().str.lower()
        
        # Map the flags and print debug info
        base_df['Flag'] = base_df['county_normalized'].map(pred_flag_map).fillna('Gray')
        print("Unique Flag values after mapping:", base_df['Flag'].unique())
        
        base_df['color'] = base_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))
        print("Unique color values:", base_df['color'].unique())
        
        base_df['prediction'] = base_df['county_normalized'].map(pred_map)

        # Add actuals for current month
        base_df['actual'] = base_df['SNAP_Applications'].apply(format_snap)
        base_df['prediction_str'] = base_df.apply(
            lambda row: f"Actual: {row['actual']}<br>Prediction for {pred_month_english}: {format_snap(row['prediction'])}", axis=1)
        filtered_df = base_df.copy()
    else:
        selected_date_dt = pd.to_datetime(selected_date)
        filtered_df = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].copy()
        snap_df['z_score'] = compute_z_scores(snap_df[snap_df['date'] <= selected_date_dt])
        zscore_map = snap_df[snap_df['date'].dt.strftime('%b %Y') == selected_date].set_index('county')['z_score'].to_dict()
        filtered_df['z_score'] = filtered_df['county'].map(zscore_map)
        filtered_df['Flag'] = filtered_df['z_score'].apply(zscore_to_flag)
        filtered_df['color'] = filtered_df['Flag'].map(lambda x: flag_color_map.get(x, "#888888"))
        filtered_df['actual'] = filtered_df['SNAP_Applications'].apply(format_snap)
        filtered_df['prediction_str'] = filtered_df.apply(
            lambda row: f"Actual: {row['actual']}<br>No prediction available", axis=1)

    # Debug: Print the columns in filtered_df
    print("Columns in filtered_df:", filtered_df.columns.tolist())
    print("Sample Flag values:", filtered_df['Flag'].head().tolist())
    print("Sample color values:", filtered_df['color'].head().tolist())
    
    fig = px.choropleth(
        filtered_df,
        geojson=counties_geojson,
        locations='county',
        color='Flag',
        color_discrete_map=flag_color_map,
        category_orders={"Flag": ["Red", "Yellow", "Green", "Gray"]},
        custom_data=['county', 'prediction_str'],
        featureidkey="properties.name",
        scope="usa",
        height=700
    )
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"
    )
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

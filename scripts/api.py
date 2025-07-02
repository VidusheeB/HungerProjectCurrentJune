from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import json
import os
from datetime import datetime
import socket
import random

app = Flask(__name__)
CORS(app)

# Define the correct data directories
base_dir = Path(__file__).parent.parent
public_dir = base_dir / 'public'
data_dir = public_dir / 'data'

# Function to find an available port
def find_available_port():
    while True:
        port = random.randint(5000, 6000)  # Try ports between 5000 and 6000
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except socket.error:
                continue

# Get an available port
PORT = find_available_port()

@app.route('/')
def index():
    return jsonify({"message": "API running", "port": PORT})

@app.route('/api/prediction/<state>', methods=['GET'])
def get_predictions(state):
    try:
        # Find the latest prediction file
        prediction_files = list(data_dir.glob('snap_predictions_*.json'))
        if not prediction_files:
            return jsonify({"error": "No prediction files found"}), 404
            
        latest_file = max(prediction_files, key=os.path.getctime)
        
        # Read the latest prediction data
        with open(latest_file, 'r') as f:
            predictions = json.load(f)
            
        # Filter predictions for the requested state
        state_predictions = [p for p in predictions if p["State"] == state]
        
        return jsonify(state_predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical/<state>', methods=['GET'])
def get_historical(state):
    try:
        # Read the dashboard data
        dashboard_path = data_dir / '2019-2023_dashboard.csv'
        if not dashboard_path.exists():
            return jsonify({"error": "Dashboard data not found"}), 404
        
        df = pd.read_csv(dashboard_path)
        # Filter data for the requested state
        state_data = df[df["State"] == state].to_dict(orient="records")
        
        return jsonify(state_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest-prediction-date', methods=['GET'])
def get_latest_prediction_date():
    try:
        # Find the latest prediction file
        prediction_files = list(data_dir.glob('snap_predictions_*.json'))
        if not prediction_files:
            return jsonify({"error": "No prediction files found"}), 404
            
        latest_file = max(prediction_files, key=os.path.getctime)
        timestamp = latest_file.stem.split('_')[-1]
        
        return jsonify({"latest_prediction_date": timestamp})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"Starting Flask server on port {PORT}")
    app.run(debug=True, port=PORT)

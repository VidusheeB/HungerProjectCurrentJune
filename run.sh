#!/bin/bash

# Start Flask server in background
source .venv/bin/activate
echo "Starting Flask server..."
python scripts/api.py &

# Wait a moment for Flask to start
sleep 2

# Start React development server in the foreground
# This will open the browser automatically
echo "Starting React development server..."
npm start

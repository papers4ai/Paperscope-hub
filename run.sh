#!/bin/bash
# Quick start script

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Fetching papers from arXiv..."
python main.py

echo ""
echo "Running processing pipeline..."
python pipeline.py

echo ""
echo "Starting local server..."
echo "Open http://localhost:8080 in your browser"
python3 -m http.server 8080
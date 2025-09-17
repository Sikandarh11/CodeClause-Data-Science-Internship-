#!/bin/bash

# CodeClause Data Science Internship Setup Script
echo "=== CodeClause Data Science Internship Setup ==="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the projects:"
echo "1. Crop Disease Detection: cd 'CNNs for identifying crop diseases' && python app.py"
echo "2. Movie Recommendation: cd 'Movie Recommendation System' && python app.py"
echo "3. Customer Segmentation: jupyter notebook"
echo "4. Time Series Forecasting: jupyter notebook"
echo ""
echo "Remember to activate the virtual environment before running:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "source venv/Scripts/activate"
else
    echo "source venv/bin/activate"
fi
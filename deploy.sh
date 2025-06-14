#!/bin/bash

# Plant Disease Detection System - Deployment Script

set -e  # Exit on any error

echo "🌱 Plant Disease Detection System - Deployment Script"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

if ! command_exists pip; then
    echo "❌ pip is required but not installed."
    exit 1
fi

echo "✅ Python and pip found"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check for model files
echo "🔍 Checking for model files..."
MODEL_FOUND=false

if [ -f "best_plant_disease_model.pth" ]; then
    echo "✅ Found: best_plant_disease_model.pth"
    MODEL_FOUND=true
elif [ -f "plant_disease_model_complete.pth" ]; then
    echo "✅ Found: plant_disease_model_complete.pth"
    MODEL_FOUND=true
elif [ -f "potato_disease_model.pth" ]; then
    echo "✅ Found: potato_disease_model.pth"
    MODEL_FOUND=true
fi

if [ "$MODEL_FOUND" = false ]; then
    echo "⚠️  No trained model found. Please add one of:"
    echo "   - best_plant_disease_model.pth"
    echo "   - plant_disease_model_complete.pth"
    echo "   - potato_disease_model.pth"
    echo ""
    echo "The app will still run but predictions won't work without a model."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p .streamlit

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the application:"
echo "   Local: streamlit run app.py"
echo "   Docker: docker-compose up --build"
echo ""
echo "🌐 The app will be available at: http://localhost:8501"
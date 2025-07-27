#!/bin/bash
# setup.sh - Setup script for the project

set -e

echo "🚀 Setting up Political Sentiment Analysis API..."

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download TextBlob corpora
echo "📖 Downloading TextBlob corpora..."
python -m textblob.download_corpora

# Create data directories
mkdir -p data/uploads data/processed logs

echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the API: ./run.sh"
echo "3. View docs: http://localhost:8000/docs"

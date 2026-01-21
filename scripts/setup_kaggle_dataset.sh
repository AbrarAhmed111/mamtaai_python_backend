#!/bin/bash

# Automated setup script for Baby Crying Sounds dataset from Kaggle
# Usage: ./scripts/setup_kaggle_dataset.sh

set -e  # Exit on error

echo "🎯 Setting up Baby Crying Sounds Dataset from Kaggle"
echo "=================================================="
echo ""

# Check if Kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle credentials not found!"
    echo "Please:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Create API Token"
    echo "  3. Place kaggle.json in ~/.kaggle/kaggle.json"
    exit 1
fi

# Step 1: Download dataset
echo "📥 Step 1: Downloading dataset from Kaggle..."
if [ -f "baby-crying-sounds-dataset.zip" ]; then
    echo "   Dataset zip already exists, skipping download..."
else
    kaggle datasets download -d baby-crying-sounds-dataset
fi

# Step 2: Extract
echo "📦 Step 2: Extracting dataset..."
if [ -d "Baby Crying Sounds" ]; then
    echo "   Dataset already extracted, skipping..."
else
    unzip -q baby-crying-sounds-dataset.zip
fi

# Step 3: Organize dataset
echo "🗂️  Step 3: Organizing dataset..."
python -m utils.dataset_download_helper \
    --map-baby-crying-sounds "Baby Crying Sounds" \
    --output baby_cry_dataset

# Step 4: Validate
echo "✅ Step 4: Validating dataset..."
python -m utils.dataset_download_helper --validate baby_cry_dataset

# Step 5: Prepare
echo "🔧 Step 5: Preparing dataset for training..."
python -m utils.dataset_preparation \
    --dataset-dir baby_cry_dataset \
    --output baby_crying_sounds_dataset.json

echo ""
echo "=================================================="
echo "✅ Dataset setup complete!"
echo ""
echo "📊 Dataset ready: baby_crying_sounds_dataset.json"
echo ""
echo "🚀 Train your model with:"
echo "   curl -X POST 'http://localhost:8000/api/classification/upload-dataset-and-train' \\"
echo "     -F 'file=@baby_crying_sounds_dataset.json'"
echo ""
echo "Or use Python:"
echo "   from utils.dataset_preparation import load_prepared_dataset"
echo "   from services.classification import BabyCryClassifier"
echo ""
echo "   training_data = load_prepared_dataset('baby_crying_sounds_dataset.json')"
echo "   classifier = BabyCryClassifier()"
echo "   classifier.train(training_data=training_data)"
echo ""

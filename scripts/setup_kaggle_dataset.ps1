# Automated setup script for Baby Crying Sounds dataset from Kaggle
# Usage: .\scripts\setup_kaggle_dataset.ps1

Write-Host "🎯 Setting up Baby Crying Sounds Dataset from Kaggle" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Kaggle is installed
try {
    $null = Get-Command kaggle -ErrorAction Stop
    Write-Host "✓ Kaggle CLI found" -ForegroundColor Green
} catch {
    Write-Host "❌ Kaggle CLI not found. Installing..." -ForegroundColor Yellow
    python -m pip install kaggle
    Write-Host "✓ Kaggle CLI installed" -ForegroundColor Green
}

# Check if kaggle.json exists
$kagglePath = "$env:USERPROFILE\.kaggle\kaggle.json"
if (-not (Test-Path $kagglePath)) {
    Write-Host "⚠️  Kaggle credentials not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To download from Kaggle, you need to:" -ForegroundColor Yellow
    Write-Host "  1. Go to https://www.kaggle.com/account" -ForegroundColor Yellow
    Write-Host "  2. Scroll to 'API' section" -ForegroundColor Yellow
    Write-Host "  3. Click 'Create New API Token'" -ForegroundColor Yellow
    Write-Host "  4. Place kaggle.json in $kagglePath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Alternatively, if you already have the dataset downloaded:" -ForegroundColor Cyan
    Write-Host "  .\scripts\setup_dataset_manual.ps1 -DatasetPath 'path/to/Baby Crying Sounds'" -ForegroundColor Gray
    Write-Host ""
    exit 1
}
Write-Host "✓ Kaggle credentials found" -ForegroundColor Green

# Step 1: Download dataset
Write-Host "📥 Step 1: Downloading dataset from Kaggle..." -ForegroundColor Yellow
if (Test-Path "baby-crying-sounds-dataset.zip") {
    Write-Host "   Dataset zip already exists, skipping download..." -ForegroundColor Gray
} else {
    kaggle datasets download -d baby-crying-sounds-dataset
}

# Step 2: Extract
Write-Host "📦 Step 2: Extracting dataset..." -ForegroundColor Yellow
if (Test-Path "Baby Crying Sounds") {
    Write-Host "   Dataset already extracted, skipping..." -ForegroundColor Gray
} else {
    Expand-Archive -Path baby-crying-sounds-dataset.zip -DestinationPath . -Force
}

# Step 3: Organize dataset
Write-Host "🗂️  Step 3: Organizing dataset..." -ForegroundColor Yellow
python -m utils.dataset_download_helper `
    --map-baby-crying-sounds "Baby Crying Sounds" `
    --output baby_cry_dataset

# Step 4: Validate
Write-Host "✅ Step 4: Validating dataset..." -ForegroundColor Yellow
python -m utils.dataset_download_helper --validate baby_cry_dataset

# Step 5: Prepare
Write-Host "🔧 Step 5: Preparing dataset for training..." -ForegroundColor Yellow
python -m utils.dataset_preparation `
    --dataset-dir baby_cry_dataset `
    --output baby_crying_sounds_dataset.json

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "✅ Dataset setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Dataset ready: baby_crying_sounds_dataset.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Train your model with:" -ForegroundColor Cyan
Write-Host "   curl -X POST 'http://localhost:8000/api/classification/upload-dataset-and-train' -F 'file=@baby_crying_sounds_dataset.json'" -ForegroundColor Gray
Write-Host ""

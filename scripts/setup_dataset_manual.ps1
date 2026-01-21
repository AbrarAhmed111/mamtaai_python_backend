# Manual setup script - works with already downloaded dataset
# Usage: .\scripts\setup_dataset_manual.ps1 -DatasetPath "path/to/Baby Crying Sounds"

param(
    [Parameter(Mandatory=$false)]
    [string]$DatasetPath = "Baby Crying Sounds"
)

Write-Host "Setting up Baby Crying Sounds Dataset" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if dataset directory exists
if (-not (Test-Path $DatasetPath)) {
    Write-Host "ERROR: Dataset directory not found: $DatasetPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please either:" -ForegroundColor Yellow
    Write-Host "  1. Download from Kaggle and extract, then run:" -ForegroundColor Yellow
    Write-Host "     .\scripts\setup_dataset_manual.ps1 -DatasetPath 'path/to/Baby Crying Sounds'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Or use the full automated script (requires Kaggle credentials):" -ForegroundColor Yellow
    Write-Host "     .\scripts\setup_kaggle_dataset.ps1" -ForegroundColor Gray
    exit 1
}

Write-Host "Dataset found at: $DatasetPath" -ForegroundColor Green
Write-Host ""

# Step 1: Organize dataset
Write-Host "Step 1: Organizing dataset..." -ForegroundColor Yellow
python -m utils.dataset_download_helper --map-baby-crying-sounds $DatasetPath --output baby_cry_dataset

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Error organizing dataset" -ForegroundColor Red
    exit 1
}

# Step 2: Validate
Write-Host "Step 2: Validating dataset..." -ForegroundColor Yellow
python -m utils.dataset_download_helper --validate baby_cry_dataset

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Validation found issues, but continuing..." -ForegroundColor Yellow
}

# Step 3: Prepare
Write-Host "Step 3: Preparing dataset for training..." -ForegroundColor Yellow
python -m utils.dataset_preparation --dataset-dir baby_cry_dataset --output baby_crying_sounds_dataset.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Error preparing dataset" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Dataset setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Dataset ready: baby_crying_sounds_dataset.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "Train your model with:" -ForegroundColor Cyan
Write-Host "  curl -X POST http://localhost:8000/api/classification/upload-dataset-and-train -F file=@baby_crying_sounds_dataset.json" -ForegroundColor Gray
Write-Host ""

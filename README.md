# MamtaAI Backend API

A FastAPI-based backend service for baby cry classification and audio processing using machine learning. This API provides endpoints for audio preprocessing, feature extraction, and ML-powered cry type prediction.

## 🎯 Features

### Audio Processing Module
- **Audio Recording Support** - Accepts multiple audio formats (WAV, MP3, M4A, OGG, WEBM)
- **Format Conversion** - Automatic conversion to standardized format
- **Noise Removal** - Advanced noise reduction using spectral gating
- **Segmentation** - Optional audio chunking for analysis
- **Normalization** - Peak and RMS normalization

### Feature Extraction Module
- **MFCC Extraction** - Mel-Frequency Cepstral Coefficients for audio analysis
- **Spectrogram Generation** - Frequency domain analysis
- **Pitch and Frequency Analysis** - Fundamental frequency detection and spectral analysis
- **Duration Analysis** - Silence detection and audio duration metrics

### Baby Cry Classification Module
- **Model Training** - Train ML models (Random Forest, Gradient Boosting) with labeled data
- **Cry Type Prediction** - Classify baby cries into categories (hungry, tired, discomfort, pain, etc.)
- **Confidence Scores** - Get prediction probabilities for all cry types
- **Continuous Model Improvement** - Retrain models with new data for better accuracy

### Streaming Audio Processing Module
- **Real-time Progress Updates** - Server-Sent Events (SSE) for live processing status
- **Complete Pipeline** - Single endpoint for preprocessing, feature extraction, and classification
- **Database Integration Ready** - Returns processed audio in base64 format for easy storage
- **Progress Tracking** - Step-by-step progress updates (preprocessing, saving, feature extraction, classification)

## 🚀 Quickstart

### Prerequisites
- Python 3.11+ (see `runtime.txt`)
- pip
- System audio libraries (for librosa/soundfile):
  - **Ubuntu/Debian**: `sudo apt-get install libsndfile1`
  - **macOS**: `brew install libsndfile`
  - **Windows**: Usually included with librosa installation

### Installation

1. **Clone and navigate to the project**
```bash
cd mamtaai_python_backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Copy the example file
cp env.example .env
```

Edit `.env`:
```env
ENVIRONMENT=development
BASE_URL=http://localhost:8000
API_VERSION=1.0.0
ALLOWED_ORIGINS=http://localhost:3000,*
PORT=8000
```

4. **Run the server**
```bash
# Option 1: Using the start script (recommended)
python start_api_server.py

# Option 2: Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

5. **Access the API**
- **API Root**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### Audio Processing

#### `POST /api/audio/process`
Complete audio processing pipeline (preprocessing + feature extraction)

**Request:**
- `file`: Audio file (multipart/form-data)
- `remove_noise`: Boolean (default: true)
- `normalize`: Boolean (default: true)
- `segment`: Boolean (default: false)
- `segment_length`: Float (default: 1.0)
- `n_mfcc`: Integer (default: 13)

**Response:**
```json
{
  "message": "Audio processed successfully",
  "processed_audio_base64": "...",
  "preprocessing": {
    "sample_rate": 22050,
    "duration": 3.5,
    "num_samples": 77175
  },
  "features": {
    "mfcc": {...},
    "spectrogram": {...},
    "pitch_frequency": {...},
    "duration": {...}
  }
}
```

#### `POST /api/audio/preprocess`
Audio preprocessing only (returns processed audio ready for DB storage)

#### `POST /api/audio/features`
Feature extraction only (returns extracted features)

#### `GET /api/audio/health`
Health check for audio processing service

### Streaming (Real-time Processing)

#### `POST /api/streaming/process-audio`
Complete audio processing pipeline with real-time progress updates via Server-Sent Events (SSE).

**Request:**
- `file`: Audio file (multipart/form-data)
- `baby_id`: String (required) - Baby ID for saving recording
- `remove_noise`: Boolean (default: true)
- `normalize`: Boolean (default: true)
- `n_mfcc`: Integer (default: 13)

**Response:** Server-Sent Events stream with progress updates:
```
data: {"step": "preprocessing", "message": "Getting audio clean format...", "progress": 0}
data: {"step": "preprocessing", "message": "Audio formatted and saved...", "progress": 100}
data: {"step": "saving", "message": "Saving cleaned audio to database...", "processed_audio_base64": "..."}
data: {"step": "feature_extraction", "message": "Extracting features...", "progress": 50}
data: {"step": "classification", "message": "Running baby cry classification...", "progress": 0}
data: {"step": "completed", "message": "Processing complete", "predicted_cry_type": "hungry", ...}
```

**Usage Example:**
```javascript
const eventSource = new EventSource(
  `http://localhost:8000/api/streaming/process-audio?file=${file}&baby_id=123&remove_noise=true`
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.step, data.message);
  if (data.step === 'completed') {
    console.log('Prediction:', data.predicted_cry_type);
  }
};
```

#### `GET /api/streaming/health`
Health check for streaming service

### Classification

#### `POST /api/classification/train`
Train a new baby cry classification model

**Request:**
```json
{
  "training_data": [
    {
      "features": {
        "mfcc": {...},
        "spectrogram": {...},
        "pitch_frequency": {...},
        "duration": {...}
      },
      "label": "hungry"
    }
  ],
  "model_type": "random_forest",
  "test_size": 0.2,
  "validation_size": 0.1,
  "cry_types": ["hungry", "tired", "discomfort", "pain"]
}
```

**Response:**
```json
{
  "message": "Model trained successfully",
  "model_path": "models/baby_cry_classifier_v1.0.0.pkl",
  "metrics": {
    "test_accuracy": 0.87,
    "test_precision": 0.85,
    "test_recall": 0.88,
    "test_f1": 0.86
  },
  "classification_report": {...},
  "confusion_matrix": [[...]]
}
```

#### `POST /api/classification/predict-from-audio`
Complete pipeline: Process audio and predict cry type

**Request:**
- `file`: Audio file (multipart/form-data)
- `remove_noise`: Boolean (default: true)
- `normalize`: Boolean (default: true)
- `n_mfcc`: Integer (default: 13)

**Response:**
```json
{
  "message": "Prediction completed successfully",
  "prediction": {
    "predicted_cry_type": "hungry",
    "confidence_score": 0.87,
    "confidence_scores": {
      "hungry": 0.87,
      "tired": 0.08,
      "discomfort": 0.03,
      "pain": 0.02
    },
    "all_predictions": [
      {"cry_type": "hungry", "confidence": 0.87},
      {"cry_type": "tired", "confidence": 0.08},
      ...
    ]
  }
}
```

#### `POST /api/classification/predict`
Predict cry type from pre-extracted features

**Request:**
```json
{
  "features": {
    "mfcc": {...},
    "spectrogram": {...},
    "pitch_frequency": {...},
    "duration": {...}
  }
}
```

#### `POST /api/classification/improve`
Continuously improve model with new training data

**Request:**
```json
{
  "new_training_data": [
    {
      "features": {...},
      "label": "hungry"
    }
  ],
  "test_size": 0.2
}
```

#### `POST /api/classification/load-model`
Load a saved model from disk

**Request:**
```json
{
  "model_path": "models/baby_cry_classifier_v1.0.0.pkl"
}
```

#### `GET /api/classification/model-info`
Get information about the current loaded model

#### `POST /api/classification/upload-dataset`
Upload a prepared dataset JSON file for training

**Request:**
- `file`: JSON file with training dataset (multipart/form-data)

**Response:**
```json
{
  "message": "Dataset uploaded successfully",
  "total_samples": 150,
  "samples_per_label": {
    "hungry": 50,
    "tired": 50,
    "pain": 50
  },
  "labels": ["hungry", "tired", "pain"]
}
```

#### `POST /api/classification/upload-dataset-and-train`
Upload a dataset and immediately train a model with it

**Request:**
- `file`: JSON file with training dataset (multipart/form-data)
- `model_type`: String (default: "random_forest")
- `test_size`: Float (default: 0.2)
- `validation_size`: Float (default: 0.1)
- `cry_types`: String (optional, comma-separated)

**Response:** Same as `/train` endpoint

#### `GET /api/classification/health`
Health check for classification service

### General Endpoints

#### `GET /`
API information and available endpoints

#### `GET /health`
Health check endpoint

## 💻 Usage Examples

### Python Example

```python
import requests

# Process audio and predict cry type
with open('baby_cry.wav', 'rb') as f:
    files = {'file': f}
    data = {
        'remove_noise': True,
        'normalize': True,
        'n_mfcc': 13
    }
    response = requests.post(
        'http://localhost:8000/api/classification/predict-from-audio',
        files=files,
        data=data
    )
    result = response.json()
    print(f"Predicted: {result['prediction']['predicted_cry_type']}")
    print(f"Confidence: {result['prediction']['confidence_score']}")
```

### JavaScript/TypeScript Example (Next.js)

**Standard Endpoint:**
```typescript
// Process audio and get prediction
const formData = new FormData();
formData.append('file', audioFile);
formData.append('remove_noise', 'true');
formData.append('normalize', 'true');

const response = await fetch('http://localhost:8000/api/classification/predict-from-audio', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Predicted cry type:', result.prediction.predicted_cry_type);
console.log('Confidence:', result.prediction.confidence_score);

// Save processed audio to database
const processedAudio = result.prediction.processed_audio_base64;
// ... save to your database
```

**Streaming Endpoint (Recommended):**
```typescript
// Process audio with real-time progress updates
const formData = new FormData();
formData.append('file', audioFile);
formData.append('baby_id', 'baby-123');
formData.append('remove_noise', 'true');
formData.append('normalize', 'true');

// Note: For SSE, you'll need to use a library like EventSource or fetch with streaming
const response = await fetch('http://localhost:8000/api/streaming/process-audio', {
  method: 'POST',
  body: formData
});

const reader = response.body?.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader!.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      console.log(data.step, data.message);
      
      if (data.step === 'saving' && data.processed_audio_base64) {
        // Save processed audio to database
        await saveToDatabase(data.processed_audio_base64, data.sample_rate, data.duration);
      }
      
      if (data.step === 'completed') {
        console.log('Prediction:', data.predicted_cry_type);
        console.log('Confidence:', data.confidence_score);
      }
    }
  }
}
```

### cURL Examples

**Standard Prediction:**
```bash
# Predict from audio file
curl -X POST "http://localhost:8000/api/classification/predict-from-audio" \
  -F "file=@baby_cry.wav" \
  -F "remove_noise=true" \
  -F "normalize=true"
```

**Streaming Endpoint:**
```bash
# Process with streaming (SSE)
curl -X POST "http://localhost:8000/api/streaming/process-audio" \
  -F "file=@baby_cry.wav" \
  -F "baby_id=baby-123" \
  -F "remove_noise=true" \
  -F "normalize=true" \
  --no-buffer
```

**Train Model:**
```bash
# Upload dataset and train
curl -X POST "http://localhost:8000/api/classification/upload-dataset-and-train" \
  -F "file=@baby_crying_sounds_dataset.json" \
  -F "model_type=random_forest" \
  -F "test_size=0.2"
```

## 📊 Dataset Preparation

### 🎯 Recommended Dataset: **Baby Crying Sounds (Kaggle)**

**Perfect dataset available on Kaggle!**
- ✅ **1,313 total files** - Large, well-distributed dataset
- ✅ **Direct matches**: hungry (382!), tired (132), discomfort (135)
- ✅ **Easy mappings**: belly pain → pain, cold_hot/burping → discomfort
- ✅ **Easy download** from Kaggle

### 🚀 Automated Setup (Recommended)

**One-command setup:**
```bash
# Linux/Mac
chmod +x scripts/setup_kaggle_dataset.sh
./scripts/setup_kaggle_dataset.sh

# Windows
.\scripts\setup_kaggle_dataset.ps1
```

This automatically:
1. Downloads dataset from Kaggle
2. Organizes and maps labels
3. Validates dataset
4. Prepares training-ready JSON

### 📝 Manual Setup

**Step 1: Download from Kaggle**
```bash
pip install kaggle
# Get API token from https://www.kaggle.com/account
kaggle datasets download -d baby-crying-sounds-dataset
unzip baby-crying-sounds-dataset.zip
```

**Step 2: Organize Dataset**
```bash
# Automated (recommended)
python -m utils.dataset_download_helper \
    --map-baby-crying-sounds "Baby Crying Sounds" \
    --output baby_cry_dataset

# Or manual organization (see docs/KAGGLE_DATASET_SETUP.md)
```

**Step 3: Prepare & Train**
```bash
python -m utils.dataset_preparation \
    --dataset-dir baby_cry_dataset \
    --output dataset.json

curl -X POST "http://localhost:8000/api/classification/upload-dataset-and-train" \
  -F "file=@dataset.json"
```

### Using Your Own Dataset

1. **Organize your audio files** by label:
   ```
   dataset/
   ├── hungry/
   │   ├── audio1.wav
   │   └── audio2.wav
   ├── tired/
   │   ├── audio1.wav
   │   └── audio2.wav
   └── ...
   ```

2. **Prepare the dataset**:
   ```bash
   python -m utils.dataset_preparation \
       --dataset-dir /path/to/your/dataset \
       --output dataset.json
   ```

3. **Upload and train**:
   ```bash
   curl -X POST "http://localhost:8000/api/classification/upload-dataset-and-train" \
     -F "file=@dataset.json"
   ```

### Documentation

- **[Quick Start Guide](DATASET_QUICK_START.md)** - Get started in 5 steps
- **[Kaggle Setup Guide](docs/KAGGLE_DATASET_SETUP.md)** - Complete Kaggle download & setup
- **[Baby Crying Sounds Integration](docs/BABY_CRYING_SOUNDS_DATASET.md)** - Detailed mapping guide
- **[Dataset Preparation Guide](docs/DATASET_PREPARATION.md)** - Technical preparation details
- **[Dataset Recommendation](docs/DATASET_RECOMMENDATION.md)** - Comparison with other datasets

## 📁 Project Structure

```
mamtaai_python_backend/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   └── routers/
│       ├── audio.py            # Audio processing endpoints
│       ├── classification.py   # ML classification endpoints
│       └── streaming.py        # Streaming audio processing with SSE
│
├── services/
│   ├── __init__.py
│   ├── audio.py                # Audio processing logic
│   └── classification.py       # ML model logic
│
├── utils/
│   ├── __init__.py
│   ├── dataset_preparation.py  # Dataset preparation utilities
│   └── dataset_download_helper.py  # Dataset download & organization helpers
│
├── scripts/
│   ├── setup_kaggle_dataset.sh      # Automated Kaggle dataset setup (Linux/Mac)
│   ├── setup_kaggle_dataset.ps1     # Automated Kaggle dataset setup (Windows)
│   └── setup_dataset_manual.ps1     # Manual dataset setup script
│
├── examples/
│   └── prepare_dataset_example.py   # Example scripts
│
├── docs/
│   ├── API_FLOW.md                  # Detailed API flow documentation
│   ├── PROJECT_STRUCTURE.md         # Project structure guide
│   ├── DATASET_PREPARATION.md       # Dataset preparation guide
│   ├── KAGGLE_DATASET_SETUP.md      # Kaggle dataset setup guide
│   ├── BABY_CRYING_SOUNDS_DATASET.md # Baby Crying Sounds dataset guide
│   └── DATASET_RECOMMENDATION.md    # Dataset comparison guide
│
├── models/                          # Saved ML models (created at runtime)
│   └── *.pkl                        # Trained classifier models
│
├── Baby Crying Sounds/              # Dataset folder (if using manual setup)
├── baby_cry_dataset/                # Organized dataset (created during setup)
├── baby_crying_sounds_dataset.json  # Prepared dataset JSON (created during setup)
│
├── config.py                        # Configuration helpers
├── requirements.txt                 # Python dependencies
├── requirements-test.txt            # Test dependencies
├── env.example                      # Environment variables template
├── Procfile                         # Railway deployment config
├── railway.json                     # Railway settings
├── runtime.txt                      # Python version (3.11)
├── start_api_server.py              # Local server startup script
├── COMMANDS.md                      # Quick command reference
├── DATASET_QUICK_START.md           # Quick dataset setup guide
└── SETUP_INSTRUCTIONS.md            # Setup instructions
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Environment (development/production) | `development` | No |
| `BASE_URL` | Base URL for the API | `http://localhost:8000` | No |
| `API_VERSION` | API version | `1.0.0` | No |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated, use `*` for all) | `*` | No |
| `PORT` | Server port | `8000` | No |
| `RAILWAY_ENVIRONMENT` | Auto-detected on Railway platform | - | Auto |

### Supported Cry Types

Default cry types for classification:
- `hungry`
- `tired`
- `discomfort`
- `pain`
- `attention`
- `diaper_change`
- `overstimulated`
- `colic`

You can customize these when training a model.

## 📚 Documentation

### Quick References
- **[Dataset Quick Start](DATASET_QUICK_START.md)** - Get started with dataset in 5 steps
- **[Setup Instructions](SETUP_INSTRUCTIONS.md)** - Complete setup guide
- **[Commands Reference](COMMANDS.md)** - Quick command reference

### Detailed Guides
- **[API Flow Documentation](docs/API_FLOW.md)** - Detailed request flow and architecture
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - File-by-file guide
- **[Dataset Preparation Guide](docs/DATASET_PREPARATION.md)** - How to prepare and add datasets
- **[Kaggle Dataset Setup](docs/KAGGLE_DATASET_SETUP.md)** - Complete Kaggle download & setup
- **[Baby Crying Sounds Integration](docs/BABY_CRYING_SOUNDS_DATASET.md)** - Detailed mapping guide
- **[Dataset Recommendation](docs/DATASET_RECOMMENDATION.md)** - Comparison with other datasets

### API Documentation
- **Interactive API Docs (Swagger)** - Available at `/docs` when server is running
- **ReDoc Documentation** - Available at `/redoc` when server is running
- **OpenAPI Schema** - Available at `/openapi.json` when server is running

## 🧪 Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests (when test suite is added)
pytest

# Run with coverage
pytest --cov=api --cov=services --cov-report=html
```

## 🚀 Deployment

### Railway

The project includes Railway configuration files:
- `Procfile` - Process definition
- `railway.json` - Railway deployment settings

1. Link your repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push

### Other Platforms

The API can be deployed to any platform that supports Python:
- Heroku
- AWS Elastic Beanstalk
- Google Cloud Run
- Azure App Service
- Docker

## 📦 Dependencies

### Core
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-dotenv` - Environment variable management

### Audio Processing
- `librosa` - Audio analysis
- `soundfile` - Audio I/O
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `noisereduce` - Noise reduction
- `pydub` - Audio manipulation

### Machine Learning
- `scikit-learn` - ML algorithms
- `joblib` - Model persistence
- `pandas` - Data manipulation

## 🔄 Workflow

### Typical Usage Flow

1. **Setup Dataset**
   ```
   → Download dataset from Kaggle (or use your own)
   → Organize and map labels using utils/dataset_download_helper
   → Prepare dataset JSON using utils/dataset_preparation
   ```

2. **Train Initial Model**
   ```
   POST /api/classification/upload-dataset-and-train
   → Upload prepared dataset JSON
   → Get trained model and metrics
   → Model saved to models/ directory
   ```

3. **Classify New Audio**
   
   **Option A: Standard Endpoint**
   ```
   POST /api/classification/predict-from-audio
   → Upload audio file
   → Get prediction with confidence scores
   → Save processed audio to database
   ```
   
   **Option B: Streaming Endpoint (Recommended for UI)**
   ```
   POST /api/streaming/process-audio
   → Upload audio file with baby_id
   → Receive real-time progress updates via SSE
   → Get prediction with processed audio base64
   → Frontend saves to database
   ```

4. **Improve Model Over Time**
   ```
   POST /api/classification/improve
   → Provide new labeled data
   → Model retrains and improves
   → Better accuracy for future predictions
   ```

## 🐛 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError` for audio libraries (librosa/soundfile)
- **Solution:** Install system audio libraries:
  - **Ubuntu/Debian**: `sudo apt-get install libsndfile1`
  - **macOS**: `brew install libsndfile`
  - **Windows**: Usually included with librosa, but if issues occur, install via conda or use WSL

**Issue:** Model not found error when predicting
- **Solution:** Train a model first using:
  ```bash
  POST /api/classification/upload-dataset-and-train
  ```
  Or check if model exists:
  ```bash
  GET /api/classification/model-info
  ```

**Issue:** CORS errors from frontend (Next.js/React)
- **Solution:** Set `ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001` in `.env` (comma-separated for multiple origins)

**Issue:** Dataset preparation takes too long
- **Solution:** This is normal! Processing hundreds of audio files takes time. The script shows progress: `[X/Total] (XX.X%)`. Be patient.

**Issue:** Streaming endpoint not working
- **Solution:** Ensure your client supports Server-Sent Events (SSE). Use `EventSource` in JavaScript or compatible SSE client.

**Issue:** Audio format not supported
- **Solution:** Supported formats: WAV, MP3, M4A, OGG, WEBM. Ensure file is not corrupted and format matches the extension.

**Issue:** Railway deployment fails
- **Solution:** 
  - Check `runtime.txt` specifies Python 3.11
  - Ensure `Procfile` is present and correct
  - Set environment variables in Railway dashboard
  - Check build logs for missing dependencies

## 🎯 Key Features Summary

- ✅ **Multi-format Audio Support** - WAV, MP3, M4A, OGG, WEBM
- ✅ **Real-time Processing** - Server-Sent Events (SSE) for progress updates
- ✅ **Advanced Audio Processing** - Noise removal, normalization, format conversion
- ✅ **Comprehensive Feature Extraction** - MFCC, spectrograms, pitch analysis
- ✅ **ML Classification** - Random Forest and Gradient Boosting models
- ✅ **Continuous Learning** - Model improvement with new data
- ✅ **Production Ready** - Railway deployment configuration included
- ✅ **Well Documented** - Comprehensive guides and API documentation

## 📝 License

This project is part of the MamtaAI application.

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure code passes linting
5. Follow Python PEP 8 style guidelines

## 📞 Support

For issues and questions:
- Check the [Troubleshooting](#-troubleshooting) section
- Review the [Documentation](#-documentation) guides
- Create an issue in the repository

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- Audio processing powered by [librosa](https://librosa.org/) and [soundfile](https://pysoundfile.readthedocs.io/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)
- Dataset: [Baby Crying Sounds Dataset](https://www.kaggle.com/datasets/baby-crying-sounds-dataset) from Kaggle

---

**Built with FastAPI ❤️ | MamtaAI Backend API**

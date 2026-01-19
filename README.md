# MamtaAI Backend API

A FastAPI-based backend service for baby cry classification and audio processing using machine learning. This API provides endpoints for audio preprocessing, feature extraction, and ML-powered cry type prediction.

## 🎯 Features

### Audio Processing Module
- **Audio Recording Support** - Accepts multiple audio formats (WAV, MP3, M4A, OGG)
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

## 🚀 Quickstart

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**
```bash
cp env.example .env
```

Edit `.env`:
```env
ENVIRONMENT=development
BASE_URL=http://localhost:8000
API_VERSION=1.0.0
ALLOWED_ORIGINS=http://localhost:3000
PORT=8000
```

3. **Run the server**
```bash
# Option 1: Using uvicorn directly
uvicorn api.main:app --reload

# Option 2: Using the start script
python start_api_server.py
```

4. **Access the API**
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

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

### cURL Example

```bash
# Predict from audio file
curl -X POST "http://localhost:8000/api/classification/predict-from-audio" \
  -F "file=@baby_cry.wav" \
  -F "remove_noise=true" \
  -F "normalize=true"
```

## 📁 Project Structure

```
mamtaai_python_backend/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   └── routers/
│       ├── audio.py            # Audio processing endpoints
│       └── classification.py   # ML classification endpoints
│
├── services/
│   ├── __init__.py
│   ├── audio.py                # Audio processing logic
│   └── classification.py       # ML model logic
│
├── models/                     # Saved ML models (created at runtime)
│   └── *.pkl                   # Trained classifier models
│
├── docs/
│   ├── API_FLOW.md            # Detailed API flow documentation
│   └── PROJECT_STRUCTURE.md   # Project structure guide
│
├── config.py                   # Configuration helpers
├── requirements.txt            # Python dependencies
├── env.example                 # Environment variables template
├── Procfile                    # Railway deployment config
├── railway.json                # Railway settings
├── runtime.txt                 # Python version
└── start_api_server.py         # Local server startup script
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/production) | `development` |
| `BASE_URL` | Base URL for the API | `http://localhost:8000` |
| `API_VERSION` | API version | `1.0.0` |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | `*` |
| `PORT` | Server port | `8000` |

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

- **[API Flow Documentation](docs/API_FLOW.md)** - Detailed request flow and architecture
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - File-by-file guide
- **Interactive API Docs** - Available at `/docs` when server is running

## 🧪 Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests (when test suite is added)
pytest
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

1. **Train Initial Model**
   ```
   POST /api/classification/train
   → Provide labeled training data
   → Get trained model and metrics
   ```

2. **Classify New Audio**
   ```
   POST /api/classification/predict-from-audio
   → Upload audio file
   → Get prediction with confidence scores
   → Save to database
   ```

3. **Improve Model Over Time**
   ```
   POST /api/classification/improve
   → Provide new labeled data
   → Model retrains and improves
   → Better accuracy for future predictions
   ```

## 🐛 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError` for audio libraries
- **Solution:** Install system audio libraries:
  - Ubuntu/Debian: `sudo apt-get install libsndfile1`
  - macOS: `brew install libsndfile`
  - Windows: Usually included with librosa

**Issue:** Model not found error
- **Solution:** Train a model first using `/api/classification/train`

**Issue:** CORS errors from Next.js
- **Solution:** Set `ALLOWED_ORIGINS=http://localhost:3000` in `.env`

## 📝 License

This project is part of the MamtaAI application.

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Ensure code passes linting

## 📞 Support

For issues and questions, please refer to the main project documentation or create an issue in the repository.

---

**Built with FastAPI ❤️**

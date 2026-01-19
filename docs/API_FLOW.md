# MamtaAI FastAPI Backend - Request Flow

## 📋 Overview

The MamtaAI backend is a FastAPI application that processes baby cry audio files and classifies them using machine learning. The system follows a clean architecture pattern with clear separation between API routes, business logic, and data processing.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│                      (api/main.py)                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              CORS Middleware                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│         ┌─────────────────┴─────────────────┐               │
│         │                                   │                │
│  ┌──────▼──────┐                  ┌────────▼────────┐      │
│  │ Audio Router│                  │Classification   │      │
│  │ /api/audio  │                  │Router            │      │
│  └──────┬──────┘                  │/api/classification│      │
│         │                          └────────┬────────┘      │
│         │                                    │               │
│  ┌──────▼──────┐                  ┌────────▼────────┐      │
│  │Audio Service│                  │Classification   │      │
│  │(services/   │                  │Service          │      │
│  │ audio.py)   │                  │(services/        │      │
│  └─────────────┘                  │ classification.py)│     │
│         │                          └─────────────────┘      │
│         │                                    │               │
│         └─────────────────┬─────────────────┘               │
│                           │                                 │
│                  ┌────────▼────────┐                       │
│                  │  ML Models      │                       │
│                  │  (models/*.pkl) │                       │
│                  └──────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Request Flow

### 1. **Entry Point: FastAPI Application** (`api/main.py`)

```
Client Request → FastAPI App → CORS Middleware → Router
```

**Steps:**
1. FastAPI app initializes with CORS middleware
2. Request comes in from Next.js frontend
3. CORS middleware validates origin
4. Request routed to appropriate router based on URL path

**Key Components:**
- `app = FastAPI()` - Main application instance
- CORS middleware configured for cross-origin requests
- Two routers mounted:
  - `/api/audio` → Audio processing
  - `/api/classification` → ML classification

---

### 2. **Audio Processing Flow** (`/api/audio/*`)

#### **Endpoint: `/api/audio/process`** (Complete Pipeline)

```
POST /api/audio/process
  │
  ├─► Receive audio file (multipart/form-data)
  │
  ├─► Read audio bytes from UploadFile
  │
  ├─► Detect format (WAV, MP3, M4A, OGG)
  │
  ├─► Call: services.audio.preprocess_audio()
  │   │
  │   ├─► convert_audio_format() → numpy array + sample_rate
  │   ├─► remove_noise() → denoised audio
  │   ├─► segment_audio() → audio chunks (optional)
  │   └─► normalize_audio() → normalized signal
  │
  ├─► Convert processed audio → WAV bytes
  │
  ├─► Call: services.audio.extract_features()
  │   │
  │   ├─► extract_mfcc() → MFCC coefficients
  │   ├─► generate_spectrogram() → frequency analysis
  │   ├─► analyze_pitch_and_frequency() → pitch metrics
  │   └─► analyze_duration() → duration metrics
  │
  └─► Return: {
        processed_audio_base64: "...",
        preprocessing: {...},
        features: {...}
      }
```

#### **Endpoint: `/api/audio/preprocess`** (Preprocessing Only)

```
POST /api/audio/preprocess
  │
  ├─► Receive audio file
  │
  ├─► Call: services.audio.preprocess_audio()
  │   (Same preprocessing steps as above)
  │
  └─► Return: {
        processed_audio_base64: "...",
        preprocessing: {...}
      }
```

#### **Endpoint: `/api/audio/features`** (Feature Extraction Only)

```
POST /api/audio/features
  │
  ├─► Receive audio file
  │
  ├─► Call: services.audio.convert_audio_format()
  │
  ├─► Call: services.audio.extract_features()
  │
  └─► Return: {
        features: {...}
      }
```

---

### 3. **Classification Flow** (`/api/classification/*`)

#### **Endpoint: `/api/classification/train`** (Model Training)

```
POST /api/classification/train
  │
  ├─► Receive: {
        training_data: [
          {features: {...}, label: "hungry"},
          {features: {...}, label: "tired"},
          ...
        ],
        model_type: "random_forest",
        test_size: 0.2
      }
  │
  ├─► Create: BabyCryClassifier(model_type)
  │
  ├─► Call: classifier.train(training_data)
  │   │
  │   ├─► Extract feature vectors from training samples
  │   ├─► Encode labels (hungry → 0, tired → 1, ...)
  │   ├─► Split data (train/validation/test)
  │   ├─► Scale features (StandardScaler)
  │   ├─► Train model (RandomForest/GradientBoosting)
  │   ├─► Evaluate on validation set
  │   ├─► Evaluate on test set
  │   └─► Calculate metrics (accuracy, precision, recall, F1)
  │
  ├─► Call: classifier.save("baby_cry_classifier")
  │   └─► Save to: models/baby_cry_classifier_v1.0.0.pkl
  │
  └─► Return: {
        model_path: "...",
        metrics: {...},
        classification_report: {...},
        confusion_matrix: [[...]]
      }
```

#### **Endpoint: `/api/classification/predict-from-audio`** (Complete Pipeline)

```
POST /api/classification/predict-from-audio
  │
  ├─► Receive audio file
  │
  ├─► Process audio (same as /api/audio/process)
  │   └─► Extract features
  │
  ├─► Call: get_model() → Load current classifier
  │
  ├─► Call: classifier.predict(features)
  │   │
  │   ├─► Extract feature vector from audio features
  │   ├─► Scale features using saved scaler
  │   ├─► Predict cry type using trained model
  │   ├─► Get prediction probabilities (confidence scores)
  │   └─► Decode label (0 → "hungry", 1 → "tired", ...)
  │
  └─► Return: {
        prediction: {
          predicted_cry_type: "hungry",
          confidence_score: 0.85,
          confidence_scores: {
            "hungry": 0.85,
            "tired": 0.10,
            ...
          },
          all_predictions: [...]
        }
      }
```

#### **Endpoint: `/api/classification/predict`** (From Features)

```
POST /api/classification/predict
  │
  ├─► Receive: {features: {...}}
  │
  ├─► Call: get_model() → Load current classifier
  │
  ├─► Call: classifier.predict(features)
  │
  └─► Return: {
        predicted_cry_type: "...",
        confidence_score: 0.85,
        ...
      }
```

#### **Endpoint: `/api/classification/improve`** (Continuous Learning)

```
POST /api/classification/improve
  │
  ├─► Receive: {
        new_training_data: [...]
      }
  │
  ├─► Call: get_model() → Get current classifier
  │
  ├─► Call: classifier.improve_with_new_data(new_data)
  │   │
  │   ├─► Retrain model with new + existing data
  │   ├─► Compare metrics (old vs new)
  │   └─► Save improved model
  │
  └─► Return: {
        metrics: {...},
        improvement: {
          accuracy_change: +0.05,
          improved: true
        }
      }
```

---

## 📦 Service Layer Details

### **Audio Service** (`services/audio.py`)

**Functions:**
- `convert_audio_format()` - Converts audio bytes to numpy array
- `remove_noise()` - Denoises audio using noisereduce
- `segment_audio()` - Splits audio into chunks
- `normalize_audio()` - Normalizes audio signal
- `extract_mfcc()` - Extracts MFCC features
- `generate_spectrogram()` - Creates spectrogram
- `analyze_pitch_and_frequency()` - Pitch analysis
- `analyze_duration()` - Duration metrics
- `extract_features()` - Complete feature extraction pipeline

### **Classification Service** (`services/classification.py`)

**BabyCryClassifier Class:**
- `train()` - Trains ML model
- `predict()` - Predicts cry type
- `save()` - Saves model to disk
- `load()` - Loads model from disk
- `improve_with_new_data()` - Continuous learning

**Model Storage:**
- Models saved in `models/` directory
- Format: `baby_cry_classifier_v{version}.pkl`
- Uses joblib for serialization

---

## 🔀 Complete End-to-End Flow Example

### **Scenario: Classify a baby cry from Next.js**

```
1. Next.js Frontend
   │
   ├─► User uploads audio file
   │
   └─► POST /api/classification/predict-from-audio
       Content-Type: multipart/form-data
       Body: {file: audio.wav}

2. FastAPI Backend
   │
   ├─► CORS middleware validates request
   │
   ├─► Router: classification.py
   │   └─► Handler: predict_from_audio()
   │
   ├─► Service: audio.py
   │   ├─► convert_audio_format() → numpy array
   │   └─► extract_features() → features dict
   │
   ├─► Service: classification.py
   │   ├─► get_model() → Load classifier
   │   └─► classifier.predict(features) → prediction
   │
   └─► Response JSON:
       {
         "prediction": {
           "predicted_cry_type": "hungry",
           "confidence_score": 0.87,
           "confidence_scores": {...}
         }
       }

3. Next.js Frontend
   │
   └─► Display prediction to user
       Save processed audio to database
```

---

## 🗂️ File Structure

```
mamtaai_python_backend/
├── api/
│   ├── main.py                 # FastAPI app, CORS, router mounting
│   └── routers/
│       ├── audio.py            # Audio processing endpoints
│       └── classification.py   # ML classification endpoints
│
├── services/
│   ├── audio.py                # Audio processing logic
│   └── classification.py       # ML model logic
│
├── models/                      # Saved ML models (created at runtime)
│   └── *.pkl                   # Trained classifier models
│
├── config.py                   # Configuration helpers
├── requirements.txt             # Python dependencies
└── docs/
    └── API_FLOW.md            # This file
```

---

## 🔑 Key Concepts

### **Separation of Concerns**
- **Routers** (`api/routers/`) - Handle HTTP requests/responses, validation
- **Services** (`services/`) - Pure business logic, no HTTP knowledge
- **Models** (`models/`) - Persisted ML models

### **Data Flow**
1. **Input**: Audio file (bytes) or features (JSON)
2. **Processing**: Audio → Features → Prediction
3. **Output**: Processed audio (base64) + Features + Prediction

### **Model Lifecycle**
1. **Train** → Create model from labeled data
2. **Save** → Persist to disk
3. **Load** → Load for predictions
4. **Predict** → Classify new audio
5. **Improve** → Retrain with new data

---

## 🚀 Usage Patterns

### **Pattern 1: Complete Pipeline** (Recommended)
```
POST /api/classification/predict-from-audio
→ Processes audio + extracts features + predicts
→ Returns everything in one call
```

### **Pattern 2: Step-by-Step**
```
1. POST /api/audio/preprocess
   → Get processed audio (save to DB)

2. POST /api/audio/features
   → Get features (save to DB)

3. POST /api/classification/predict
   → Get prediction (save to DB)
```

### **Pattern 3: Training Workflow**
```
1. Collect labeled audio samples
2. Extract features for each sample
3. POST /api/classification/train
   → Train model, get metrics
4. Use model for predictions
5. Periodically: POST /api/classification/improve
   → Retrain with new data
```

---

## 📊 Response Formats

### **Audio Processing Response**
```json
{
  "message": "Audio processed successfully",
  "processed_audio_base64": "UklGRiQ...",
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

### **Classification Response**
```json
{
  "prediction": {
    "predicted_cry_type": "hungry",
    "confidence_score": 0.87,
    "confidence_scores": {
      "hungry": 0.87,
      "tired": 0.08,
      "discomfort": 0.03,
      ...
    },
    "all_predictions": [
      {"cry_type": "hungry", "confidence": 0.87},
      {"cry_type": "tired", "confidence": 0.08},
      ...
    ]
  }
}
```

---

## 🔧 Environment Variables

```env
ENVIRONMENT=development
BASE_URL=http://localhost:8000
API_VERSION=1.0.0
ALLOWED_ORIGINS=http://localhost:3000
PORT=8000
```

---

## 📝 Notes

- All audio is converted to WAV format for processing
- Models are saved with version numbers for tracking
- Feature extraction uses librosa for audio analysis
- Classification uses scikit-learn (RandomForest/GradientBoosting)
- CORS is configured to allow Next.js frontend requests
- Models directory is created automatically on first save

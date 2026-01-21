"""
Baby Cry Classification Router
Handles model training, prediction, and continuous improvement endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import json
import io

from services.classification import (
    BabyCryClassifier,
    get_model,
    set_model,
    DEFAULT_CRY_TYPES
)
from services.audio import extract_features, convert_audio_format

router = APIRouter(prefix="/classification", tags=["classification"])


class TrainingSample(BaseModel):
    """Single training sample."""
    features: Dict = Field(..., description="Extracted audio features")
    label: str = Field(..., description="Cry type label")


class TrainingRequest(BaseModel):
    """Request model for training."""
    training_data: List[TrainingSample] = Field(..., description="List of training samples")
    model_type: str = Field("random_forest", description="Model type: 'random_forest' or 'gradient_boosting'")
    test_size: float = Field(0.2, ge=0.0, le=0.5, description="Proportion of data for testing")
    validation_size: float = Field(0.1, ge=0.0, le=0.5, description="Proportion of data for validation")
    cry_types: Optional[List[str]] = Field(None, description="List of cry type labels")


class TrainingResponse(BaseModel):
    """Response model for training."""
    message: str
    model_path: str
    metrics: Dict
    classification_report: Dict
    confusion_matrix: List[List[int]]


class PredictionRequest(BaseModel):
    """Request model for prediction from features."""
    features: Dict = Field(..., description="Extracted audio features")


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    predicted_cry_type: str
    confidence_score: float
    confidence_scores: Dict[str, float]
    all_predictions: List[Dict[str, float]]


class ImprovementRequest(BaseModel):
    """Request model for continuous improvement."""
    new_training_data: List[TrainingSample] = Field(..., description="New training samples")
    test_size: float = Field(0.2, ge=0.0, le=0.5, description="Proportion of new data for testing")


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Train a baby cry classification model.
    
    Requires training data with extracted features and labels.
    """
    try:
        # Create classifier
        classifier = BabyCryClassifier(
            model_type=request.model_type,
            cry_types=request.cry_types or DEFAULT_CRY_TYPES
        )
        
        # Prepare training data
        training_data = [
            {
                "features": sample.features,
                "label": sample.label
            }
            for sample in request.training_data
        ]
        
        # Train model
        result = classifier.train(
            training_data=training_data,
            test_size=request.test_size,
            validation_size=request.validation_size
        )
        
        # Save model
        model_name = "baby_cry_classifier"
        model_path = classifier.save(model_name)
        
        # Set as current model
        set_model(classifier, model_path)
        
        return TrainingResponse(
            message="Model trained successfully",
            model_path=model_path,
            metrics=result["metrics"],
            classification_report=result["classification_report"],
            confusion_matrix=result["confusion_matrix"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
async def predict_from_features(request: PredictionRequest):
    """
    Predict cry type from extracted audio features.
    
    Requires pre-extracted features from audio processing.
    """
    try:
        # Get current model
        classifier = get_model()
        
        # Predict
        result = classifier.predict(request.features)
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@router.post("/predict-from-audio")
async def predict_from_audio(
    file: UploadFile = File(..., description="Audio file to classify"),
    remove_noise: bool = Form(True, description="Remove noise from audio"),
    normalize: bool = Form(True, description="Normalize audio signal"),
    n_mfcc: int = Form(13, description="Number of MFCC coefficients")
):
    """
    Complete pipeline: Process audio and predict cry type.
    
    This endpoint:
    1. Processes the audio (preprocessing + feature extraction)
    2. Predicts the cry type using the trained model
    3. Returns prediction with confidence scores
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Get file format
        content_type = file.content_type or ""
        input_format = "wav"
        if "mp3" in content_type.lower() or (file.filename and file.filename.endswith(".mp3")):
            input_format = "mp3"
        elif "m4a" in content_type.lower() or (file.filename and file.filename.endswith(".m4a")):
            input_format = "m4a"
        elif "ogg" in content_type.lower() or (file.filename and file.filename.endswith(".ogg")):
            input_format = "ogg"
        
        # Convert audio format
        audio, sample_rate = convert_audio_format(audio_bytes, input_format)
        
        # Extract features
        features = extract_features(audio, sample_rate, n_mfcc=n_mfcc)
        
        # Get current model
        classifier = get_model()
        
        # Predict
        prediction_result = classifier.predict(features)
        
        return {
            "message": "Prediction completed successfully",
            "prediction": prediction_result,
            "features_extracted": {
                "mfcc_coefficients": len(features.get("mfcc", {}).get("mfcc_mean", [])),
                "has_spectrogram": "spectrogram" in features,
                "has_pitch_analysis": "pitch_frequency" in features,
                "has_duration_analysis": "duration" in features
            }
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@router.post("/improve")
async def improve_model(request: ImprovementRequest):
    """
    Continuously improve the model with new training data.
    
    This implements incremental learning by retraining the model
    with additional data to improve accuracy over time.
    """
    try:
        # Get current model
        classifier = get_model()
        
        # Prepare new training data
        new_training_data = [
            {
                "features": sample.features,
                "label": sample.label
            }
            for sample in request.new_training_data
        ]
        
        # Improve model
        result = classifier.improve_with_new_data(
            new_training_data=new_training_data,
            test_size=request.test_size
        )
        
        # Save improved model
        model_name = "baby_cry_classifier"
        model_path = classifier.save(model_name)
        
        # Set as current model
        set_model(classifier, model_path)
        
        return {
            "message": "Model improved successfully",
            "model_path": model_path,
            "metrics": result["metrics"],
            "improvement": result.get("improvement", {}),
            "classification_report": result.get("classification_report", {})
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error improving model: {str(e)}")


@router.post("/load-model")
async def load_model(model_path: str = Body(..., embed=True)):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the model file (.pkl)
    """
    try:
        classifier = BabyCryClassifier.load(model_path)
        set_model(classifier, model_path)
        
        return {
            "message": "Model loaded successfully",
            "model_path": model_path,
            "model_type": classifier.model_type,
            "cry_types": classifier.cry_types,
            "metrics": classifier.training_metrics,
            "version": classifier.version
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@router.get("/model-info")
async def get_model_info():
    """Get information about the current model."""
    try:
        classifier = get_model()
        
        return {
            "model_type": classifier.model_type,
            "cry_types": classifier.cry_types,
            "version": classifier.version,
            "metrics": classifier.training_metrics,
            "num_features": len(classifier.feature_names) if classifier.feature_names else 0,
            "model_loaded": classifier.model is not None
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(..., description="JSON file with training dataset")):
    """
    Upload a prepared dataset JSON file for training.
    
    The JSON file should contain a list of training samples:
    [
        {
            "features": {...},
            "label": "hungry"
        },
        ...
    ]
    
    Returns the dataset statistics and allows you to train with it.
    """
    try:
        # Read and parse JSON file
        content = await file.read()
        dataset = json.loads(content.decode('utf-8'))
        
        if not isinstance(dataset, list):
            raise HTTPException(status_code=400, detail="Dataset must be a JSON array")
        
        # Validate dataset structure
        valid_samples = []
        for i, sample in enumerate(dataset):
            if not isinstance(sample, dict):
                continue
            if "features" not in sample or "label" not in sample:
                continue
            valid_samples.append(sample)
        
        if len(valid_samples) == 0:
            raise HTTPException(status_code=400, detail="No valid training samples found in dataset")
        
        # Count samples per label
        label_counts = {}
        for sample in valid_samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            "message": "Dataset uploaded successfully",
            "total_samples": len(valid_samples),
            "samples_per_label": label_counts,
            "labels": list(label_counts.keys()),
            "next_step": "Use /api/classification/train endpoint with this dataset"
        }
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")


@router.post("/upload-dataset-and-train")
async def upload_dataset_and_train(
    file: UploadFile = File(..., description="JSON file with training dataset"),
    model_type: str = Form("random_forest", description="Model type: 'random_forest' or 'gradient_boosting'"),
    test_size: float = Form(0.2, description="Proportion of data for testing"),
    validation_size: float = Form(0.1, description="Proportion of data for validation"),
    cry_types: Optional[str] = Form(None, description="Comma-separated list of cry types (optional)")
):
    """
    Upload a dataset and immediately train a model with it.
    
    This is a convenience endpoint that combines dataset upload and training.
    """
    try:
        # Read and parse JSON file
        content = await file.read()
        dataset = json.loads(content.decode('utf-8'))
        
        if not isinstance(dataset, list):
            raise HTTPException(status_code=400, detail="Dataset must be a JSON array")
        
        # Prepare training data
        training_data = []
        for sample in dataset:
            if not isinstance(sample, dict):
                continue
            if "features" not in sample or "label" not in sample:
                continue
            training_data.append({
                "features": sample["features"],
                "label": sample["label"]
            })
        
        if len(training_data) == 0:
            raise HTTPException(status_code=400, detail="No valid training samples found in dataset")
        
        # Parse cry types if provided
        cry_types_list = None
        if cry_types:
            cry_types_list = [ct.strip() for ct in cry_types.split(",")]
        
        # Create classifier
        classifier = BabyCryClassifier(
            model_type=model_type,
            cry_types=cry_types_list or DEFAULT_CRY_TYPES
        )
        
        # Train model
        result = classifier.train(
            training_data=training_data,
            test_size=test_size,
            validation_size=validation_size
        )
        
        # Save model
        model_name = "baby_cry_classifier"
        model_path = classifier.save(model_name)
        
        # Set as current model
        set_model(classifier, model_path)
        
        return TrainingResponse(
            message="Dataset uploaded and model trained successfully",
            model_path=model_path,
            metrics=result["metrics"],
            classification_report=result["classification_report"],
            confusion_matrix=result["confusion_matrix"]
        )
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@router.get("/health")
async def classification_health():
    """Health check for classification service."""
    try:
        classifier = get_model()
        return {
            "status": "healthy",
            "service": "baby-cry-classification",
            "model_available": classifier.model is not None,
            "model_type": classifier.model_type,
            "endpoints": {
                "train": "/api/classification/train - Train new model",
                "predict": "/api/classification/predict - Predict from features",
                "predict-from-audio": "/api/classification/predict-from-audio - Complete pipeline",
                "improve": "/api/classification/improve - Improve existing model",
                "load-model": "/api/classification/load-model - Load saved model",
                "upload-dataset": "/api/classification/upload-dataset - Upload prepared dataset",
                "upload-dataset-and-train": "/api/classification/upload-dataset-and-train - Upload and train"
            }
        }
    except ValueError:
        return {
            "status": "no_model",
            "service": "baby-cry-classification",
            "message": "No model loaded. Train a model first.",
            "endpoints": {
                "train": "/api/classification/train - Train new model",
                "upload-dataset": "/api/classification/upload-dataset - Upload prepared dataset",
                "upload-dataset-and-train": "/api/classification/upload-dataset-and-train - Upload and train"
            }
        }


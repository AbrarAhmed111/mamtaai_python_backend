"""
Audio Processing Router
Handles audio upload, preprocessing, and feature extraction endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import Response
from typing import Optional
from pydantic import BaseModel
import io

from services.audio import preprocess_audio, extract_features, convert_audio_format

router = APIRouter(prefix="/audio", tags=["audio"])


class AudioProcessingResponse(BaseModel):
    """Response model for audio processing."""
    message: str
    processed_audio_base64: Optional[str] = None
    preprocessing: dict
    features: dict


class AudioPreprocessingResponse(BaseModel):
    """Response model for audio preprocessing only."""
    message: str
    processed_audio_base64: Optional[str] = None
    preprocessing: dict


@router.post("/process", response_model=AudioProcessingResponse)
async def process_audio(
    file: UploadFile = File(..., description="Audio file to process"),
    remove_noise: bool = Form(True, description="Remove noise from audio"),
    normalize: bool = Form(True, description="Normalize audio signal"),
    segment: bool = Form(False, description="Segment audio into chunks"),
    segment_length: float = Form(1.0, description="Length of segments in seconds"),
    n_mfcc: int = Form(13, description="Number of MFCC coefficients to extract")
):
    """
    Complete audio processing pipeline:
    1. Preprocess audio (format conversion, noise removal, segmentation, normalization)
    2. Extract features (MFCC, spectrogram, pitch/frequency analysis, duration analysis)
    
    Returns processed audio and extracted features.
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Get file format from content type or filename
        content_type = file.content_type or ""
        input_format = "wav"
        if "mp3" in content_type.lower() or file.filename.endswith(".mp3"):
            input_format = "mp3"
        elif "m4a" in content_type.lower() or file.filename.endswith(".m4a"):
            input_format = "m4a"
        elif "ogg" in content_type.lower() or file.filename.endswith(".ogg"):
            input_format = "ogg"
        
        # Step 1: Preprocess audio
        preprocessing_result = preprocess_audio(
            audio_data=audio_bytes,
            input_format=input_format,
            remove_noise_flag=remove_noise,
            normalize_flag=normalize,
            segment_flag=segment,
            segment_length_seconds=segment_length
        )
        
        # Step 2: Extract features from processed audio
        # Convert processed audio bytes back to numpy array for feature extraction
        processed_audio, sample_rate = convert_audio_format(
            preprocessing_result["processed_audio"],
            input_format="wav"
        )
        
        features = extract_features(processed_audio, sample_rate, n_mfcc=n_mfcc)
        
        # Convert processed audio to base64 for easy transmission
        import base64
        processed_audio_base64 = base64.b64encode(preprocessing_result["processed_audio"]).decode('utf-8')
        
        return AudioProcessingResponse(
            message="Audio processed successfully",
            processed_audio_base64=processed_audio_base64,
            preprocessing=preprocessing_result,
            features=features
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@router.post("/preprocess", response_model=AudioPreprocessingResponse)
async def preprocess_audio_endpoint(
    file: UploadFile = File(..., description="Audio file to preprocess"),
    remove_noise: bool = Form(True, description="Remove noise from audio"),
    normalize: bool = Form(True, description="Normalize audio signal"),
    segment: bool = Form(False, description="Segment audio into chunks"),
    segment_length: float = Form(1.0, description="Length of segments in seconds")
):
    """
    Audio preprocessing only:
    - Format Conversion
    - Noise Removal
    - Segmentation (optional)
    - Normalization
    
    Returns processed audio ready for database storage.
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Get file format from content type or filename
        content_type = file.content_type or ""
        input_format = "wav"
        if "mp3" in content_type.lower() or (file.filename and file.filename.endswith(".mp3")):
            input_format = "mp3"
        elif "m4a" in content_type.lower() or (file.filename and file.filename.endswith(".m4a")):
            input_format = "m4a"
        elif "ogg" in content_type.lower() or (file.filename and file.filename.endswith(".ogg")):
            input_format = "ogg"
        
        # Preprocess audio
        preprocessing_result = preprocess_audio(
            audio_data=audio_bytes,
            input_format=input_format,
            remove_noise_flag=remove_noise,
            normalize_flag=normalize,
            segment_flag=segment,
            segment_length_seconds=segment_length
        )
        
        # Convert processed audio to base64 for easy transmission
        import base64
        processed_audio_base64 = base64.b64encode(preprocessing_result["processed_audio"]).decode('utf-8')
        
        return AudioPreprocessingResponse(
            message="Audio preprocessed successfully",
            processed_audio_base64=processed_audio_base64,
            preprocessing=preprocessing_result
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing audio: {str(e)}")


@router.post("/features")
async def extract_audio_features(
    file: UploadFile = File(..., description="Audio file to extract features from"),
    n_mfcc: int = Form(13, description="Number of MFCC coefficients to extract")
):
    """
    Extract features from audio:
    - MFCC Extraction
    - Spectrogram Generation
    - Pitch And Frequency Analysis
    - Duration Analysis
    
    Returns extracted features only (no audio preprocessing).
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Get file format from content type or filename
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
        
        return {
            "message": "Features extracted successfully",
            "features": features
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")


@router.get("/health")
async def audio_health():
    """Health check for audio processing service."""
    return {
        "status": "healthy",
        "service": "audio-processing",
        "endpoints": {
            "process": "/api/audio/process - Complete processing pipeline",
            "preprocess": "/api/audio/preprocess - Preprocessing only",
            "features": "/api/audio/features - Feature extraction only"
        }
    }


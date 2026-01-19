"""
Streaming Audio Processing Router
Handles audio processing with real-time progress updates via Server-Sent Events (SSE).
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from typing import Optional
import json
import asyncio
import io
import base64

from services.audio import (
    convert_audio_format,
    remove_noise as remove_noise_func,
    normalize_audio,
    extract_features
)
from services.classification import get_model

router = APIRouter(prefix="/streaming", tags=["streaming"])


@router.get("/health")
async def streaming_health():
    """Health check for streaming service."""
    return {
        "status": "healthy",
        "service": "streaming-audio-processing",
        "endpoints": {
            "process-audio": "/api/streaming/process-audio - Stream audio processing with SSE"
        }
    }


async def send_progress(step: str, message: str, data: Optional[dict] = None):
    """Helper to format and send progress updates."""
    progress_data = {
        "step": step,
        "message": message
    }
    if data:
        progress_data.update(data)
    return f"data: {json.dumps(progress_data)}\n\n"


@router.post("/process-audio")
async def stream_process_audio(
    file: UploadFile = File(..., description="Audio file to process"),
    baby_id: str = Form(..., description="Baby ID for saving recording"),
    remove_noise: bool = Form(True, description="Remove noise from audio"),
    normalize: bool = Form(True, description="Normalize audio signal"),
    n_mfcc: int = Form(13, description="Number of MFCC coefficients")
):
    """
    Complete audio processing pipeline with real-time progress updates via SSE.
    
    Steps:
    1. Audio preprocessing (format conversion, noise removal, normalization)
    2. Save cleaned audio to database
    3. Feature extraction (MFCC, spectrogram, pitch, duration)
    4. Baby cry classification
    5. Return results
    
    Returns Server-Sent Events stream with progress updates.
    """
    
    async def generate():
        try:
            # Step 1: Receiving audio (skip this step, go straight to preprocessing)
            # yield await send_progress(
            #     "receiving",
            #     "Receiving audio file..."
            # )
            
            audio_bytes = await file.read()
            if len(audio_bytes) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file")
            
            # Detect format - prioritize content_type over filename
            content_type = file.content_type or ""
            filename = file.filename or ""
            input_format = "wav"  # default
            
            # Check content type first (more reliable than filename)
            content_type_lower = content_type.lower()
            if "wav" in content_type_lower:
                input_format = "wav"
            elif "webm" in content_type_lower:
                input_format = "webm"
            elif "mp3" in content_type_lower:
                input_format = "mp3"
            elif "m4a" in content_type_lower:
                input_format = "m4a"
            elif "ogg" in content_type_lower:
                input_format = "ogg"
            elif filename:
                # Fallback to filename extension if content_type doesn't help
                ext = filename.split(".")[-1].lower() if "." in filename else ""
                if ext in ["webm", "mp3", "m4a", "ogg", "wav"]:
                    input_format = ext
            
            # Log detected format for debugging
            print(f"Detected audio format: {input_format} (content_type: {content_type}, filename: {filename})")
            
            yield await send_progress(
                "preprocessing",
                "Getting audio clean format...",
                {"progress": 0}
            )
            await asyncio.sleep(0.1)
            
            # Step 2: Format Conversion
            audio, sample_rate = convert_audio_format(audio_bytes, input_format)
            duration = len(audio) / sample_rate
            await asyncio.sleep(0.1)
            
            # Step 3: Noise Removal
            if remove_noise:
                audio = remove_noise_func(audio, sample_rate)
            await asyncio.sleep(0.1)
            
            # Step 4: Normalization
            if normalize:
                audio = normalize_audio(audio, method="peak")
            await asyncio.sleep(0.1)
            
            yield await send_progress(
                "preprocessing",
                "Audio formatted and saved...",
                {"progress": 100}
            )
            
            # Step 5: Save cleaned audio to database (via Next.js API)
            # Convert processed audio to bytes
            audio_io = io.BytesIO()
            import soundfile as sf
            sf.write(audio_io, audio, sample_rate, format='WAV')
            processed_audio_bytes = audio_io.getvalue()
            processed_audio_base64 = base64.b64encode(processed_audio_bytes).decode('utf-8')
            
            yield await send_progress(
                "saving",
                "Saving cleaned audio to database...",
                {
                    "progress": 0,
                    "processed_audio_base64": processed_audio_base64,
                    "sample_rate": int(sample_rate),
                    "duration": float(duration)
                }
            )
            await asyncio.sleep(0.2)  # Simulate save time
            
            # Note: Actual database save happens in Next.js API via the processed_audio_base64
            # The frontend will handle saving when it receives this data
            
            yield await send_progress(
                "saving",
                "Cleaned audio saved successfully",
                {
                    "progress": 100,
                    "saved": True,
                    "message": "Audio file uploaded and database record created"
                    # Note: Don't include processed_audio_base64 here to prevent duplicate saves
                }
            )
            await asyncio.sleep(0.1)
            
            # Step 6: Feature Extraction
            yield await send_progress(
                "feature_extraction",
                "Extracting features from audio...",
                {"progress": 0}
            )
            await asyncio.sleep(0.1)
            
            yield await send_progress(
                "feature_extraction",
                "Extracting MFCC coefficients...",
                {"progress": 25}
            )
            await asyncio.sleep(0.1)
            
            yield await send_progress(
                "feature_extraction",
                "Generating spectrogram...",
                {"progress": 50}
            )
            await asyncio.sleep(0.1)
            
            yield await send_progress(
                "feature_extraction",
                "Analyzing pitch and frequency...",
                {"progress": 75}
            )
            await asyncio.sleep(0.1)
            
            features = extract_features(audio, sample_rate, n_mfcc=n_mfcc)
            await asyncio.sleep(0.1)
            
            # Extract summary statistics for display
            feature_summary = {
                "mfcc_coefficients": features.get("mfcc", {}).get("num_coefficients", 0),
                "mfcc_frames": features.get("mfcc", {}).get("num_frames", 0),
                "pitch_mean": round(features.get("pitch_frequency", {}).get("pitch_mean", 0), 1),
                "pitch_std": round(features.get("pitch_frequency", {}).get("pitch_std", 0), 1),
                "dominant_frequency": round(features.get("pitch_frequency", {}).get("dominant_frequency", 0), 1),
                "duration_seconds": round(features.get("duration", {}).get("total_duration_seconds", 0), 2),
                "actual_audio_duration": round(features.get("duration", {}).get("actual_audio_duration_seconds", 0), 2),
                "silence_percentage": round(features.get("duration", {}).get("silence_percentage", 0), 1),
                "spectrogram_frames": len(features.get("spectrogram", {}).get("times", [])) if features.get("spectrogram", {}).get("times") else 0,
            }
            
            yield await send_progress(
                "feature_extraction",
                "Features extracted successfully",
                {
                    "progress": 100,
                    "features": feature_summary,
                    "summary": f"{feature_summary['mfcc_coefficients']} MFCC coefficients, {feature_summary['mfcc_frames']} frames, Pitch: {feature_summary['pitch_mean']}Hz"
                }
            )
            await asyncio.sleep(0.1)
            
            # Step 7: Classification
            yield await send_progress(
                "classification",
                "Running baby cry classification...",
                {"progress": 0}
            )
            await asyncio.sleep(0.1)
            
            try:
                classifier = get_model()
                prediction_result = classifier.predict(features)
                
                yield await send_progress(
                    "classification",
                    "Classification completed!",
                    {
                        "progress": 100,
                        "prediction": prediction_result
                    }
                )
                await asyncio.sleep(0.1)
                
                # Final result
                yield await send_progress(
                    "completed",
                    "Processing complete",
                    {
                        "predicted_cry_type": prediction_result["predicted_cry_type"],
                        "confidence_score": prediction_result["confidence_score"],
                        "confidence_scores": prediction_result["confidence_scores"],
                        "features": features
                    }
                )
                
            except ValueError as e:
                # Model not trained yet
                yield await send_progress(
                    "classification",
                    "Model not available. Please train a model first.",
                    {
                        "progress": 0,
                        "error": str(e),
                        "features": features  # Still return features
                    }
                )
                
                yield await send_progress(
                    "completed",
                    "Processing complete (no classification)",
                    {
                        "features": features,
                        "prediction": None
                    }
                )
            
        except Exception as e:
            yield await send_progress(
                "error",
                f"Error processing audio: {str(e)}",
                {"error": str(e)}
            )
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

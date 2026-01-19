"""
MamtaAI Backend API (FastAPI)
Baby cry classification and audio processing API.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from api.routers.audio import router as audio_router
from api.routers.classification import router as classification_router
from api.routers.streaming import router as streaming_router

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",")]
    if ALLOWED_ORIGINS_STR and ALLOWED_ORIGINS_STR != "*"
    else ["*"]
)

# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="MamtaAI Backend API",
    description="""
    ## MamtaAI Backend API
    
    A comprehensive FastAPI-based backend service for baby cry classification and audio processing using machine learning.
    
    ### Features
    
    #### 🎵 Audio Processing
    - **Multi-format Support**: Accepts WAV, MP3, M4A, OGG formats
    - **Noise Reduction**: Advanced spectral gating for noise removal
    - **Audio Normalization**: Peak and RMS normalization
    - **Segmentation**: Optional audio chunking for analysis
    
    #### 🔬 Feature Extraction
    - **MFCC Extraction**: Mel-Frequency Cepstral Coefficients
    - **Spectrogram Generation**: Frequency domain analysis
    - **Pitch Analysis**: Fundamental frequency detection
    - **Duration Metrics**: Silence detection and audio duration analysis
    
    #### 🤖 Machine Learning
    - **Model Training**: Train Random Forest or Gradient Boosting classifiers
    - **Cry Type Prediction**: Classify baby cries into multiple categories
    - **Confidence Scores**: Get prediction probabilities for all cry types
    - **Continuous Improvement**: Retrain models with new data
    
    ### Cry Types
    - `hungry` - Baby is hungry
    - `tired` - Baby is tired
    - `discomfort` - General discomfort
    - `pain` - Pain or distress
    - `attention` - Seeking attention
    - `diaper_change` - Needs diaper change
    - `overstimulated` - Overstimulated
    - `colic` - Colic symptoms
    
    ### API Documentation
    - **Swagger UI**: Interactive API documentation at `/docs`
    - **ReDoc**: Alternative documentation at `/redoc`
    - **OpenAPI Schema**: JSON schema at `/openapi.json`
    """,
    version=API_VERSION,
    terms_of_service="https://mamtaai.com/terms",
    contact={
        "name": "MamtaAI Support",
        "url": "https://mamtaai.com/support",
        "email": "support@mamtaai.com",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://mamtaai.com/license",
    },
    servers=[
        {
            "url": os.getenv("BASE_URL", "http://localhost:8000"),
            "description": "Development server" if ENVIRONMENT == "development" else "Production server",
        },
    ],
    tags_metadata=[
        {
            "name": "audio",
            "description": "Audio processing endpoints. Handle audio upload, preprocessing, format conversion, and feature extraction.",
        },
        {
            "name": "classification",
            "description": "Baby cry classification endpoints. Train ML models, make predictions, and improve model accuracy.",
        },
        {
            "name": "streaming",
            "description": "Streaming audio processing endpoints with real-time progress updates via Server-Sent Events (SSE).",
        },
    ],
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    is_railway = os.getenv("RAILWAY_ENVIRONMENT", "") != ""
    return {
        "message": "MamtaAI Backend API",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "base_url": base_url,
        "deployment": "Railway" if is_railway else "Local",
        "docs": f"{base_url}/docs",
        "health": f"{base_url}/health",
        "endpoints": {
            "audio": f"{base_url}/api/audio",
            "classification": f"{base_url}/api/classification",
            "streaming": f"{base_url}/api/streaming",
            "streaming_health": f"{base_url}/api/streaming/health",
            "streaming_process": f"{base_url}/api/streaming/process-audio",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "mamtaai-backend",
        "version": API_VERSION,
        "environment": ENVIRONMENT,
    }


# Custom OpenAPI schema for enhanced Swagger UI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    tags_metadata = [
        {
            "name": "audio",
            "description": "Audio processing endpoints. Handle audio upload, preprocessing, format conversion, and feature extraction.",
        },
        {
            "name": "classification",
            "description": "Baby cry classification endpoints. Train ML models, make predictions, and improve model accuracy.",
        },
    ]
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=tags_metadata,
        servers=app.servers,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Mount routers
app.include_router(audio_router, prefix="/api")
app.include_router(classification_router, prefix="/api")
app.include_router(streaming_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")

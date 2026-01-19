"""
MamtaAI Backend API (FastAPI)
Baby cry classification and audio processing API.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers.audio import router as audio_router
from api.routers.classification import router as classification_router

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",")]
    if ALLOWED_ORIGINS_STR and ALLOWED_ORIGINS_STR != "*"
    else ["*"]
)

# Create FastAPI app
app = FastAPI(
    title="MamtaAI Backend API",
    description="Baby cry classification and audio processing API with ML capabilities.",
    version=API_VERSION,
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


# Mount routers
app.include_router(audio_router, prefix="/api")
app.include_router(classification_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")

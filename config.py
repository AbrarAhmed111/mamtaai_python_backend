import os

# Generic environment configuration for the template
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_VERSION = os.getenv("API_VERSION", "1.0.0")

# CORS configuration
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",")]
    if ALLOWED_ORIGINS_STR and ALLOWED_ORIGINS_STR != "*"
    else ["*"]
)

# Platform detection
IS_RAILWAY = os.getenv("RAILWAY_ENVIRONMENT", "") != ""


def get_base_url() -> str:
    """Return base URL for the service."""
    return BASE_URL


def get_config() -> dict:
    """Return sanitized configuration for introspection endpoints."""
    return {
        "environment": ENVIRONMENT,
        "base_url": get_base_url(),
        "api_version": API_VERSION,
        "allowed_origins": ALLOWED_ORIGINS,
        "is_railway": IS_RAILWAY,
    }

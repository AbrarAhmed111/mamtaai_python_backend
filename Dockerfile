FROM python:3.11-slim

# Install ffmpeg and required system libs for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal production requirements and install
COPY requirements-prod.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-prod.txt

# Optional: wav2vec2 inference support (CPU-only torch keeps the image lean).
# On Railway, set service variable INSTALL_WAV2VEC2=true to enable, plus
# WAV2VEC2_HF_REPO / HF_TOKEN so weights download from HuggingFace Hub at startup.
ARG INSTALL_WAV2VEC2=false
RUN if [ "$INSTALL_WAV2VEC2" = "true" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir transformers huggingface_hub; \
    fi

# Copy application
COPY . .

ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

EXPOSE 8000

# Railway injects $PORT at runtime — bind uvicorn to it
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

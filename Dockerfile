FROM python:3.11-slim

# System libs for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential && \
    rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces runs the container as non-root user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Install production deps + wav2vec2 inference stack (CPU-only torch keeps it lean)
COPY --chown=user requirements-prod.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-prod.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers huggingface_hub

# Copy application
COPY --chown=user . .

ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
# HuggingFace Spaces expects the app on port 7860 (see app_port in README)
ENV PORT=7860
# HF model cache lives in the user's home so it is writable
ENV HF_HOME=/home/user/.cache/huggingface

USER user

EXPOSE 7860

# start_api_server.py reads $PORT via os.getenv — no shell expansion needed
CMD ["python", "start_api_server.py"]

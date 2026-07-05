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

# Copy application
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use the Python entrypoint which reads $PORT at runtime
CMD ["python", "start_api_server.py"]
FROM python:3.11-slim

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# System deps for librosa / soundfile / scipy
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY --chown=user . .

# HuggingFace Spaces expects port 7860
ENV PORT=7860
ENV ENVIRONMENT=production

USER user

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

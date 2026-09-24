# Production Dockerfile for LiveKit Voice Agent Worker
FROM python:3.11-slim

# Install system dependencies (ffmpeg for audio streams)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the backend folder contents to /app
COPY backend/ .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Environment optimizations for production container
ENV PYTHONUNBUFFERED=1 \
    LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Pre-download required LiveKit agent model files
RUN python -m livekit.agents download-files

EXPOSE 10000

# Run LiveKit agent worker in production mode
CMD ["python", "app.py", "start"]

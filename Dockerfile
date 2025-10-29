FROM nvidia/cuda:12.8.1-base-ubuntu24.04

# System deps
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      git ffmpeg libsndfile1 curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 appuser
USER appuser
WORKDIR /app

# Create and use a dedicated virtualenv
ENV VENV_PATH=/app/.venv
RUN python3 -m venv "$VENV_PATH"
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy requirements and install into the venv
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Create models directory
RUN mkdir -p /app/models

# Copy app
COPY --chown=appuser:appuser . /app

# Runtime env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose API port (uses 8005 per app)
EXPOSE 8005

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://localhost:8005/ || exit 1

# Start server via entrypoint showing CUDA info
ENTRYPOINT ["./entrypoint.sh"]

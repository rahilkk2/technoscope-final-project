# ══════════════════════════════════════════════════════════════════════════════
# TechnoScope — Production Dockerfile (CPU)
# Multi-stage: Python 3.11 base + Node.js 20 LTS
# Replace python:3.11-slim with nvidia/cuda:12.2.0-runtime-ubuntu22.04
# if GPU inference is needed.
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim AS base

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js 20 LTS ───────────────────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
# CPU-only PyTorch keeps the image ~1.5 GB smaller than full CUDA build
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        opencv-python-headless \
        onnxruntime \
        numpy \
        Pillow \
        pywt \
        scikit-image \
        scipy \
        exifread

WORKDIR /app

# ── Node dependencies (cached layer — only rebuilds when package files change)
COPY backend/package.json backend/package-lock.json ./backend/
RUN cd backend && npm ci --omit=dev

# ── Application source ───────────────────────────────────────────────────────
COPY backend/          ./backend/
COPY aranged.html      ./aranged.html

# ── Model weights (volume-mountable at runtime) ──────────────────────────────
# Default: baked-in weights. Override via docker-compose volume.
COPY backend/models/   ./backend/models/

# ── Runtime setup ─────────────────────────────────────────────────────────────
# Create dirs the app expects at runtime
RUN mkdir -p backend/uploads backend/projects

EXPOSE 8000

# Health check — container marked unhealthy if server is down
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

WORKDIR /app/backend
CMD ["node", "server.js"]

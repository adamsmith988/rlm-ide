# =============================================================================
# Tool Runner Service Dockerfile
# سرویس اجرای ابزارها - نسخه مستقل AIv2
# Build from repo root:
#   docker build -f services/tool-runner/Dockerfile -t aiv2/tool-runner:dev .
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# bubblewrap: OS jail for RLM IPython workers (mount namespace). Requires the
# container/pod to allow user namespaces (e.g. securityContext privileged, or
# CAP_SYS_ADMIN + seccomp unconfined). Set AIV2_RLM_REQUIRE_BWRAP=1 in prod when
# the runtime supports it; Landlock inside the worker is the fallback FS boundary.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools for native Python wheels (yara-python, rawpy, …)
    build-essential \
    pkg-config \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Audio processing (soundfile / ffmpeg probes in file_meta, plugins)
    ffmpeg \
    libsndfile1 \
    # File type detection (python-magic)
    libmagic1 \
    # Extended metadata (pyexiftool used by file_meta)
    libimage-exiftool-perl \
    # yara-python / rawpy native libs
    libyara-dev \
    libraw-dev \
    # RLM worker OS jail
    bubblewrap \
    # Utilities
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY services/tool-runner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service source code
COPY services/tool-runner/src/ ./src/
COPY packages/aiv2-model-contract/ /app/packages/aiv2-model-contract/

# Create directories
RUN mkdir -p /data/models /data/uploads /app/modules

# Environment variables
ENV PYTHONPATH="/app:/app/packages/aiv2-model-contract/src:/app/modules"
ENV TOOLS_MODULE_PATH="/app/modules/tools"
ENV MODELS_DIR="/data/models"
ENV UPLOAD_DIR="/data/models"
# Prefer bwrap when the pod allows user namespaces; Landlock still jails FS.
ENV AIV2_RLM_BWRAP="1"
ENV AIV2_RLM_LANDLOCK="1"
# Set to 1 only when the runtime is known to support bwrap (e.g. privileged).
ENV AIV2_RLM_REQUIRE_BWRAP="0"

EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the service. Single process only (no --workers: agent-flow state is
# process-local). --timeout-graceful-shutdown bounds how long uvicorn waits for
# open connections (long-lived SSE) on SIGTERM so the lifespan drain hook
# (agent_flow_drain) always runs within terminationGracePeriodSeconds.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002", "--timeout-graceful-shutdown", "30"]

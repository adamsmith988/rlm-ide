# Build from repo root:
#   docker build -f services/storage/Dockerfile -t aiv2/storage:dev .
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY services/storage/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY services/storage/src/ ./src/
COPY packages/aiv2-model-contract/ /app/packages/aiv2-model-contract/

# Create data directories
RUN mkdir -p /app/data/uploads

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/packages/aiv2-model-contract/src

# Expose port
EXPOSE 8004

# Run
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8004"]

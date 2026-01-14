# Backend Dockerfile for Mapping Heat
FROM python:3.9-slim

# Install system dependencies for LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model.py .
COPY app.py .
COPY db.py .
COPY stats.py .
COPY schema.sql .

# Create directories
RUN mkdir -p /app/artifacts /app/instance

# Copy artifacts directory
COPY artifacts/ ./artifacts/

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run Flask app
CMD ["python", "-u", "app.py"]
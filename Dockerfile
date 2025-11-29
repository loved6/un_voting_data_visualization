# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-docker.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create dataset directory
RUN mkdir -p /app/dataset

# Copy source code
COPY src/ /app/src/
COPY dataset/ /app/dataset/

# Copy any other necessary files
COPY README.md /app/
COPY LICENSE /app/

# Expose the port that Dash uses by default
EXPOSE 8050

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Change to src directory and run the app
WORKDIR /app/src
CMD ["python", "app.py"]
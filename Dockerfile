FROM python:3.10-slim

# Suppress TensorFlow oneDNN warnings
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements-hf.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-hf.txt

# Copy application files
COPY app.py .
COPY model_info.json .
COPY best_autoencoder_model.h5 .
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY test_images/ ./test_images/

# Hugging Face Spaces runs on port 7860
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
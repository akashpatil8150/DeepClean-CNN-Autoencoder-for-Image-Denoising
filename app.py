"""
Flask Web Application for Image Denoising
"""
import os
import sys
import numpy as np
import json
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained model
MODEL_PATH = 'best_autoencoder_model.h5'
MODEL_INFO_PATH = 'model_info.json'
model = None
model_info = None

def load_trained_model():
    """Load the trained autoencoder model"""
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found!")

def load_model_info():
    """Load model information from JSON file"""
    global model_info
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)
        print(f"Model info loaded from {MODEL_INFO_PATH}")
    else:
        # Default info if file doesn't exist
        model_info = {
            "model_name": "CNN Autoencoder",
            "architecture": "Convolutional Autoencoder",
            "test_accuracy": "N/A",
            "test_f1_score": "N/A",
            "test_loss": "N/A"
        }
        print(f"Warning: Model info file {MODEL_INFO_PATH} not found! Using defaults.")

def preprocess_image(image):
    """Preprocess uploaded image for model"""
    # Convert to grayscale
    img = image.convert('L')
    # Resize to 28x28
    img = img.resize((28, 28))
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def array_to_base64(img_array):
    """Convert numpy array to base64 string for display"""
    # Remove batch and channel dimensions
    img_array = img_array.squeeze()
    # Convert to 0-255 range
    img_array = (img_array * 255).astype(np.uint8)
    # Create PIL image
    img = Image.fromarray(img_array, mode='L')
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Load model and info at module level so it works with Docker/gunicorn
load_trained_model()
load_model_info()


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', model_info=model_info)

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Return model information"""
    if model_info:
        return jsonify(model_info)
    return jsonify({'error': 'Model info not available'}), 404

@app.route('/denoise', methods=['POST'])
def denoise():
    """Handle image denoising request"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read and preprocess image
        image = Image.open(file.stream)
        processed_img = preprocess_image(image)
        
        # Denoise image
        denoised_img = model.predict(processed_img, verbose=0)
        
        # Convert to base64 for display
        original_b64 = array_to_base64(processed_img)
        denoised_b64 = array_to_base64(denoised_img)
        
        return jsonify({
            'original': original_b64,
            'denoised': denoised_b64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Hugging Face Spaces requires port 7860; fallback to 5000 for local dev
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)

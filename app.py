import os, sys, numpy as np, json, io, base64
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from flask import Flask, render_template, request, jsonify
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODEL_PATH = "best_autoencoder_model.h5"
MODEL_INFO_PATH = "model_info.json"
SAMPLES_DIR = "test_images"
model = None
model_info = None


def load_trained_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: {MODEL_PATH} not found")
        return
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    try:
        inp = Input(shape=(28, 28, 1))
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
        enc = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(16, (3, 3), activation="relu", padding="same")(enc)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        out = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
        rebuilt = Model(inp, out)
        rebuilt.load_weights(MODEL_PATH)
        model = rebuilt
        print("Model loaded via weights-only approach")
    except Exception as e:
        print(f"Model load failed: {e}")


def load_model_info():
    global model_info
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH) as f:
            model_info = json.load(f)
    else:
        model_info = {
            "model_name": "CNN Autoencoder",
            "architecture": "Convolutional Autoencoder",
            "test_accuracy": "N/A",
            "test_f1_score": "N/A",
            "test_loss": "N/A"
        }


def preprocess_image(image):
    img = image.convert("L").resize((28, 28))
    arr = np.array(img) / 255.0
    return arr.reshape(1, 28, 28, 1)


def array_to_base64(arr):
    arr = arr.squeeze()
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


load_trained_model()
load_model_info()


@app.route("/")
def index():
    return render_template("index.html", model_info=model_info)


@app.route("/api/model-info")
def get_model_info():
    if model_info:
        return jsonify(model_info)
    return jsonify({"error": "not available"}), 404


@app.route("/api/samples")
def get_samples():
    """Return list of sample images as base64 thumbnails"""
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(SAMPLES_DIR, fname)
                with open(fpath, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                # Parse label from filename e.g. noisy_digit_2_8.png -> Digit 2
                parts = fname.replace(".png", "").split("_")
                label = f"Digit {parts[2]}" if len(parts) >= 3 else fname
                samples.append({
                    "filename": fname,
                    "label": label,
                    "thumbnail": f"data:image/png;base64,{b64}"
                })
    return jsonify(samples)


@app.route("/api/denoise-sample", methods=["POST"])
def denoise_sample():
    """Denoise a built-in sample image by filename"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.get_json()
    filename = data.get("filename", "")
    # Sanitize: only allow filenames, no path traversal
    filename = os.path.basename(filename)
    fpath = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "Sample not found"}), 404
    try:
        image = Image.open(fpath)
        proc = preprocess_image(image)
        denoised = model.predict(proc, verbose=0)
        return jsonify({"original": array_to_base64(proc), "denoised": array_to_base64(denoised)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/denoise", methods=["POST"])
def denoise():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No image selected"}), 400
    try:
        image = Image.open(file.stream)
        proc = preprocess_image(image)
        denoised = model.predict(proc, verbose=0)
        return jsonify({"original": array_to_base64(proc), "denoised": array_to_base64(denoised)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)

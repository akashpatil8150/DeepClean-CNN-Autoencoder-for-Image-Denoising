---
title: DeepClean CNN Autoencoder For Image Denoising
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
short_description: CNN Autoencoder image denoiser trained on MNIST
---

<div align="center">

# 🧹 DeepClean
### CNN Autoencoder for Image Denoising

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-FF4B4B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Akash8150/DeepClean-CNN-Autoencoder-for-Image-Denoising)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

A deep learning web app that removes noise from handwritten digit images using a **Convolutional Autoencoder** trained on the MNIST dataset.

</div>

---

## ✨ Features

- 📤 Upload any grayscale image — auto-resized to 28×28
- 🔍 Side-by-side comparison of noisy vs. denoised output
- 🖼️ One-click sample images for instant testing
- ⚡ Fast inference with a lightweight CNN model
- 🐳 Dockerized and deployed on Hugging Face Spaces

---

## 🚀 Live Demo

> Try it instantly — no setup required!

**👉 [https://huggingface.co/spaces/Akash8150/DeepClean-CNN-Autoencoder-for-Image-Denoising](https://huggingface.co/spaces/Akash8150/DeepClean-CNN-Autoencoder-for-Image-Denoising)**

---

## 🧠 How It Works

Upload a noisy grayscale image and the model reconstructs a clean version through an encoder-decoder pipeline.

```
Input (28×28)
     │
     ▼
┌─────────────────────────────┐
│         ENCODER             │
│  Conv2D(32) → MaxPool       │
│  Conv2D(16) → MaxPool       │
│  Latent Space: 7×7×16       │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│         DECODER             │
│  Conv2D(16) → UpSample      │
│  Conv2D(32) → UpSample      │
│  Conv2D(1, sigmoid)         │
└─────────────────────────────┘
     │
     ▼
Output (28×28) — Clean Image
```

---

## 📊 Model Performance

| Metric         | Value   |
|----------------|---------|
| ✅ Test Accuracy | 87.56%  |
| 📈 F1 Score      | 0.8923  |
| 📉 Test Loss     | 0.1234  |

---

## 🗂️ Dataset

| Property        | Details                          |
|-----------------|----------------------------------|
| Dataset         | MNIST Handwritten Digits         |
| Training Samples| 60,000                           |
| Test Samples    | 10,000                           |
| Noise Type      | Gaussian (factor = 0.5)          |

---

## 🏗️ Project Structure

```
DeepClean/
├── app.py                      # Flask web server
├── best_autoencoder_model.h5   # Trained model weights
├── model_info.json             # Model metadata
├── Dockerfile                  # Container config
├── requirements-hf.txt         # Dependencies
├── src/
│   ├── model.py                # Model architecture
│   ├── data_loader.py          # Data pipeline
│   └── utils.py                # Helper functions
├── static/
│   ├── script.js               # Frontend logic
│   └── style.css               # Styling
├── templates/
│   └── index.html              # Web UI
└── test_images/                # Sample noisy digits
```

---

## 🛠️ Tech Stack

| Layer      | Technology              |
|------------|-------------------------|
| Model      | TensorFlow / Keras      |
| Backend    | Flask                   |
| Image Proc | Pillow (PIL)            |
| Deployment | Docker + Hugging Face   |

---

## 🏃 Run Locally

```bash
# Clone the repo
git clone https://github.com/akashpatil8150/DeepClean-CNN-Autoencoder-for-Image-Denoising.git
cd DeepClean-CNN-Autoencoder-for-Image-Denoising

# Install dependencies
pip install -r requirements-hf.txt

# Start the app
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Using Docker

```bash
docker build -t deepclean .
docker run -p 7860:7860 deepclean
```

---

<div align="center">

Made with ❤️ by [Akash Patil](https://github.com/akashpatil8150)

</div>

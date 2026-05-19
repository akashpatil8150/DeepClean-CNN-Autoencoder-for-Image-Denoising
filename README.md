---
title: DeepClean CNN Autoencoder For Image Denoising
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
short_description: AI-powered image denoiser using a Convolutional Autoencoder trained on MNIST
---

#  DeepClean  CNN Autoencoder for Image Denoising

A deep learning web app that removes noise from handwritten digit images using a **Convolutional Autoencoder** trained on the MNIST dataset.

## How It Works

Upload a noisy grayscale image (any size  it gets resized to 28x28 automatically), and the model reconstructs a clean version.

### Model Architecture

- **Encoder**: Conv2D(32) -> MaxPool -> Conv2D(16) -> MaxPool -> latent space (7x7x16)
- **Decoder**: Conv2D(16) -> UpSample -> Conv2D(32) -> UpSample -> Conv2D(1, sigmoid)

### Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 87.56% |
| F1 Score | 0.8923 |
| Test Loss | 0.1234 |

### Dataset

- **MNIST** Handwritten Digits
- 60,000 training samples / 10,000 test samples
- Gaussian noise (factor = 0.5) added during training

## Tech Stack

- TensorFlow / Keras
- Flask
- Pillow
- Docker (Hugging Face Spaces)

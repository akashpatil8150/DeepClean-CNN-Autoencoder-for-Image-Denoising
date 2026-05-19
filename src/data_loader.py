"""
Data Loader Module
Handles loading and preprocessing of MNIST dataset with noise addition
"""
import numpy as np
from tensorflow.keras.datasets import mnist


def load_and_preprocess_data():
    """
    Load MNIST dataset and preprocess images
    
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """
    # Load MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()
    
    # Normalize pixel values to range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to (samples, height, width, channels) for CNN
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    
    return (x_train, x_train), (x_test, x_test)


def add_noise(images, noise_factor=0.5):
    """
    Add Gaussian noise to images
    
    Gaussian noise is random noise with normal distribution.
    This simulates real-world image corruption.
    
    Args:
        images: Clean images array
        noise_factor: Amount of noise to add (default: 0.5)
    
    Returns:
        Noisy images clipped to valid range [0, 1]
    """
    # Generate random Gaussian noise with same shape as images
    noise = np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    
    # Add noise to images
    noisy_images = images + noise_factor * noise
    
    # Clip values to ensure they stay in valid range [0, 1]
    noisy_images = np.clip(noisy_images, 0.0, 1.0)
    
    return noisy_images

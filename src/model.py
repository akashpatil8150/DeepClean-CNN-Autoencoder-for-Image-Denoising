"""
Model Architecture Module
Defines the Convolutional Autoencoder architecture
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D


def build_autoencoder():
    """
    Build Convolutional Autoencoder model for image denoising
    
    Autoencoder: Neural network that learns to compress (encode) and 
    reconstruct (decode) data. Used here to learn clean image representation.
    
    Architecture:
    - Encoder: Compresses noisy image to latent representation
    - Latent Space: Compressed representation capturing essential features
    - Decoder: Reconstructs clean image from latent representation
    
    Returns:
        Compiled Keras model
    """
    # Input layer: 28x28 grayscale images
    input_img = Input(shape=(28, 28, 1))
    
    # ========== ENCODER ==========
    # Encoder compresses input image to lower-dimensional latent space
    # This forces model to learn essential features while removing noise
    
    # Conv2D: Convolutional layer extracts spatial features using filters
    # - 32 filters learn different patterns (edges, textures)
    # - 3x3 kernel size for local feature detection
    # - ReLU activation introduces non-linearity
    # - padding='same' maintains spatial dimensions
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    
    # MaxPooling2D: Downsamples by taking maximum value in 2x2 window
    # Reduces spatial dimensions from 28x28 to 14x14
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Second convolutional block with fewer filters (16)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    
    # Further downsample from 14x14 to 7x7
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # ========== LATENT SPACE ==========
    # At this point: 7x7x16 = 784 values (compressed from 28x28 = 784 pixels)
    # Latent space captures essential image features without noise
    
    # ========== DECODER ==========
    # Decoder reconstructs clean image from compressed representation
    
    # Convolutional layer to process latent features
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    
    # UpSampling2D: Increases spatial dimensions by repeating values
    # Upsamples from 7x7 to 14x14
    x = UpSampling2D((2, 2))(x)
    
    # Expand feature maps back to 32 filters
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Upsample from 14x14 to 28x28 (original size)
    x = UpSampling2D((2, 2))(x)
    
    # Final layer: 1 filter to produce single-channel grayscale output
    # Sigmoid activation ensures output values in range [0, 1]
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Create model mapping input to decoded output
    autoencoder = Model(input_img, decoded)
    
    return autoencoder


def compile_model(model):
    """
    Compile the autoencoder model
    
    Args:
        model: Keras model to compile
    
    Returns:
        Compiled model
    """
    # Adam optimizer: Adaptive learning rate optimization algorithm
    # binary_crossentropy: Measures difference between predicted and actual pixel values
    # accuracy: Tracks how close predictions are to targets
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

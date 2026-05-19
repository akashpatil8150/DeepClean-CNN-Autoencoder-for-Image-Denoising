"""
Utility Module
Visualization and helper functions
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Keras training history object
    """
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plots saved as 'training_history.png'")
    plt.show()


def visualize_results(model, noisy_images, clean_images, num_images=5):
    """
    Display comparison of noisy, original, and denoised images
    
    Args:
        model: Trained autoencoder model
        noisy_images: Noisy input images
        clean_images: Original clean images
        num_images: Number of images to display (default: 5)
    """
    # Generate denoised predictions
    denoised_images = model.predict(noisy_images[:num_images])
    
    # Create figure with 3 rows (Noisy, Original, Denoised) and num_images columns
    fig, axes = plt.subplots(3, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        # Row 1: Noisy images
        axes[0, i].imshow(noisy_images[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Noisy Input', fontsize=12, fontweight='bold')
        
        # Row 2: Original clean images
        axes[1, i].imshow(clean_images[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Original Clean', fontsize=12, fontweight='bold')
        
        # Row 3: Denoised output from model
        axes[2, i].imshow(denoised_images[i].reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Denoised Output', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('denoising_results.png', dpi=300, bbox_inches='tight')
    print("✓ Denoising results saved as 'denoising_results.png'")
    plt.show()


def print_model_summary(model):
    """
    Print detailed model architecture summary
    
    Args:
        model: Keras model
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    model.summary()
    print("="*60 + "\n")

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Standard ImageNet normalization values required by PyTorch models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def getSpatialTransform(train=True):
    """Transforms for the standard, visible image."""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def getDFT(img):
    """
    Computes the Discrete Fourier Transform (DFT) of an image to expose 
    hidden manipulation artifacts (like copy-move or deepfake artifacts).
    """
    # 1. Convert to grayscale (frequency analysis only needs intensity)
    gray = img.convert('L')
    img_np = np.array(gray)
    
    # 2. Compute 2D Fast Fourier Transform
    f_transform = np.fft.fft2(img_np)
    
    # 3. Shift the zero-frequency component to the center
    f_shift = np.fft.fftshift(f_transform)
    
    # 4. Calculate magnitude spectrum (logarithmic scale)
    # Add a tiny epsilon (1e-8) to avoid math domain errors (log of 0)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
    
    # 5. Normalize the array to 0-255 so it can be treated as a standard image
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / \
                         (np.max(magnitude_spectrum) - np.min(magnitude_spectrum) + 1e-8) * 255.0
                         
    # Convert back to PIL Image (RGB format so it can be processed by standard CNNs)
    return Image.fromarray(magnitude_spectrum.astype(np.uint8)).convert('RGB')

def getFrequencyTransform(train=True):
    """Transforms for the Fourier Transform image."""
    # We do not use RandomFlip here because frequency space is highly sensitive to orientation changes
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
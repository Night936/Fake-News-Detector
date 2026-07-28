import numpy as np
import torch
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
    Computes the log-magnitude Discrete Fourier Transform (DFT) of an image to expose
    hidden manipulation artifacts (like copy-move or deepfake artifacts).

    This is the branch already wired into Models/imageModel.py's ImageExpertsModel /
    imageTraining.py -- kept unchanged so that pipeline keeps working as-is.
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

def getFrequencyComponents(img, size=224):
    """
    Real+imaginary DFT features for the progressive fusion network (Models/progressiveFusion.py).

    Jing et al. (2023, Section 3.2) explicitly keep the imaginary (phase) part of the DFT
    separate from the real/amplitude part before feeding VGG19 -- their ablation shows
    dropping phase costs ~0.6% accuracy on Weibo. getDFT() above collapses everything into
    a single log-magnitude image and throws phase away, so it can't be reused here.

    Returns a 6-channel tensor: [R_real, G_real, B_real, R_imag, G_imag, B_imag], each
    channel independently log-compressed and standardized (raw DFT coefficients have a
    huge dynamic range that would blow up a downstream conv net otherwise). The paper
    doesn't specify the exact real/imaginary concatenation shape it fed into VGG19 (which
    natively expects 3 channels), so this 6-channel layout is a pragmatic, documented
    choice -- Models/progressiveFusion.py adapts it back down to 3 channels with a learned
    1x1 conv before VGG19, keeping VGG19's pretrained weights intact.
    """
    rgb = img.convert('RGB').resize((size, size))
    arr = np.array(rgb).astype(np.float32)  # (H, W, 3)

    def _standardize(x):
        # sign-preserving log compression, then zero-mean/unit-variance per channel
        x = np.sign(x) * np.log1p(np.abs(x))
        mean, std = x.mean(), x.std() + 1e-8
        return (x - mean) / std

    real_channels, imag_channels = [], []
    for c in range(3):
        f = np.fft.fftshift(np.fft.fft2(arr[:, :, c]))
        real_channels.append(_standardize(f.real))
        imag_channels.append(_standardize(f.imag))

    stacked = np.stack(real_channels + imag_channels, axis=0)  # (6, H, W)
    return torch.tensor(stacked, dtype=torch.float32)

def getFrequencyTransform(train=True):
    """Transforms for the Fourier Transform image (magnitude-only branch, unchanged)."""
    # We do not use RandomFlip here because frequency space is highly sensitive to orientation changes
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
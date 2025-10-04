import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import pandas as pd
import numpy as np
import os

def preprocess_image(image_path, label):
    """Loads, resizes, converts to grayscale, and preprocesses an image."""
    # base_image_dir = '/content/dataset/Training/' # Assuming images are in the training directory structure
    # full_image_path = os.path.join(base_image_dir, image_path)

    img = cv2.imread(image_path)

    if img is None:
        print(f"Warning: Could not load image from {image_path}")
        return None, label # Return None for image if loading fails

    # Resize image (e.g., to 128x128)
    img_resized = cv2.resize(img, (256, 128))

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Apply denoising (e.g., using non-local means denoising) - Reverted to a simpler method or removed based on user feedback
    # img_denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_denoised = cv2.fastNlMeansDenoising(img_gray, None, 10, 7, 21) # Example of another denoising method

    # Apply thresholding (if desired)
    _, thresholded = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)


    # Normalize pixel values
    img_normalized = thresholded / 255.0 # Normalize the grayscale image


    return img_normalized, label



def display_img(im_path):
  dpi = 80
  im_data = plt.imread(im_path)
  height, width = im_data.shape[:2]
  figsize = width / float(dpi), height / float(dpi)
  fig = plt.figure(figsize=figsize)
  ax = fig.add_axes([0, 0, 1, 1])
  ax.axis('off')
  ax.imshow(im_data, cmap='gray')
  plt.show()



def apply_preprocessing_to_row(row, base_image_dir):
    """Applies preprocessing to a single row of the DataFrame and returns a tuple."""
    image_path = os.path.join(base_image_dir, row['IMAGE'])
    label = row['MEDICINE_NAME'] # Or 'GENERIC_NAME' depending on which label is needed
    processed_image, processed_label = preprocess_image(image_path, label)
    return (processed_image, processed_label) # Return as a tuple



def image_to_tensor(image_array, target_height=32, max_width=256):
    """
    Preprocesses an image array for CRNN input using PyTorch operations.

    Args:
        image_array: NumPy array of the image.
        target_height: The target height for the image.
        max_width: The maximum width for the image.

    Returns:
        A preprocessed image tensor with a consistent height and padded width.
    """
    # Convert NumPy array to PyTorch tensor
    img_tensor = torch.from_numpy(image_array).float()

    # Ensure image is grayscale with a channel dimension (C, H, W)
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0) # Add channel dimension for grayscale
    elif img_tensor.shape[-1] == 3:
        img_tensor = img_tensor.permute(2, 0, 1) # Change HWC to CHW
        img_tensor = F.rgb_to_grayscale(img_tensor) # Convert to grayscale

    original_height = img_tensor.shape[1]
    original_width = img_tensor.shape[2]

    # Calculate new width while maintaining aspect ratio
    new_width = int(original_width * target_height / original_height)

    # Resize the image
    img_resized = F.resize(img_tensor, size=[target_height, new_width])

    # Pad or crop the width
    padding_width = max_width - new_width
    if padding_width > 0:
        # Pad the image on the right side
        img_padded = F.pad(img_resized, padding=[0, 0, padding_width, 0])
    else:
        # Crop the image if the new width exceeds max_width
        img_padded = img_resized[:, :, :max_width]


    return img_padded

# Create a character to integer mapping and vice versa
# Get all unique characters from the labels


def create_ctc_labels(label, char_to_int, max_label_length=None):
    """
    Creates CTC-friendly labels from a text label.

    Args:
        label: The text label.
        char_to_int: Mapping from character to integer.
        max_label_length: The maximum length for padding (optional).

    Returns:
        A list of integers representing the label, padded if max_label_length is provided.
    """
    ctc_label = [char_to_int[char] for char in str(label) if char in char_to_int]
    if max_label_length:
        # Pad with a special character if needed (e.g., a character not in the vocabulary)
        # Here, we assume the padding character's integer representation is len(char_to_int)
        padding_value = len(char_to_int) # Use an integer outside the valid character range for padding
        ctc_label = ctc_label + [padding_value] * (max_label_length - len(ctc_label))
    return ctc_label


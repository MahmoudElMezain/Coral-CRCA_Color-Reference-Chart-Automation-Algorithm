import numpy as np
import cv2

def getTransmission(img, AtomsphericLight, blockSize):
    """
    Efficient computation of transmission map using GB channels.

    Args:
        img (np.ndarray): BGR input image
        AtomsphericLight (tuple/list): Estimated atmospheric light [B, G, R]
        blockSize (int): Size of local patch

    Returns:
        np.ndarray: Refined transmission map, values in [0.1, 0.9]
    """
    # Normalize the blue and green channels by their corresponding A values
    norm_b = img[:, :, 0] / AtomsphericLight[0]
    norm_g = img[:, :, 1] / AtomsphericLight[1]

    # Get per-pixel min of (B, G)
    norm_min = np.minimum(norm_b, norm_g)

    # Apply min filter (erode)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (blockSize, blockSize))
    dark_channel = cv2.erode(norm_min, kernel)

    # Transmission map
    transmission = 1 - dark_channel

    # Clip to safe range
    transmission = np.clip(transmission, 0.1, 0.9)

    return transmission

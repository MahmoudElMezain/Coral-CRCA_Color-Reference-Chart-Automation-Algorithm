import numpy as np
import cv2

def getDarkChannel(img, blockSize):
    """
    Efficient UDCP dark channel computation using OpenCV.

    Args:
        img (np.ndarray): BGR image
        blockSize (int): size of the patch (must be odd)

    Returns:
        np.ndarray: dark channel image (uint8)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input must be a BGR image")

    # Step 1: Take min between Blue and Green channels
    gb_min = np.minimum(img[:, :, 0], img[:, :, 1])  # B and G channels only

    # Step 2: Erode using a block kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (blockSize, blockSize))
    dark_channel = cv2.erode(gb_min, kernel)

    return dark_channel
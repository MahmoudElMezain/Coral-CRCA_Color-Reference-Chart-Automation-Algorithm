import numpy as np

def getAtomsphericLight(darkChannel, img):
    """
    Efficient version of atmospheric light estimation.

    Picks the pixel with the highest value in the dark channel
    and returns its corresponding RGB value from the original image.

    Args:
        darkChannel (np.ndarray): 2D dark channel map
        img (np.ndarray): Original BGR image

    Returns:
        np.ndarray: Atmospheric light (B, G, R)
    """
    # Flatten the dark channel to find max value location
    flat_index = np.argmax(darkChannel)
    h, w = darkChannel.shape
    y, x = np.unravel_index(flat_index, (h, w))

    # Return the BGR pixel at that location
    return img[y, x, :]

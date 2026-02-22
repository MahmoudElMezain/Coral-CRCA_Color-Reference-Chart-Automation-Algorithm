import numpy as np
import cv2

from Denoising_UTILS.UDCP.GuidedFilter import GuidedFilter
from Denoising_UTILS.UDCP.getGbDarkChannel import getDarkChannel
from Denoising_UTILS.UDCP.getAtomsphericLight import getAtomsphericLight
from Denoising_UTILS.UDCP.getTM import getTransmission
from Denoising_UTILS.UDCP.RefinedTramsmission import Refinedtransmission
from Denoising_UTILS.UDCP.sceneRadiance import sceneRadianceRGB


class ColorCorrection:
    def __init__(self, algorithm_name="WB_GRAY", blockSize=3):
        """
        Initializes the ColorCorrection module.

        Args:
            algorithm_name (str): One of ["WB_GRAY", "WB_RGB_MAX", "DCP"]
        """
        self.algorithm_name = algorithm_name.upper()
        self.blockSize = blockSize
        
    def apply(self, image):
        """
        Applies the selected color correction algorithm to the input image.

        Args:
            image (np.ndarray): Input image in BGR format

        Returns:
            np.ndarray: Color corrected image in BGR format
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy ndarray (e.g., from cv2.imread).")

        if self.algorithm_name == "WB_GRAY":
            return self._gray_world_white_balance(image)
        elif self.algorithm_name == "WB_RGB_MAX":
            return self._max_rgb_white_balance(image)
        elif self.algorithm_name == "DCP":
            return self._dcp_dehaze(image)
        elif self.algorithm_name == "UDCP":
            return self._udcp_dehaze(image)
        else:
            raise ValueError(f"Unsupported algorithm name: {self.algorithm_name}")

    def _gray_world_white_balance(self, image):
        img = image.astype(np.float32)
        avg_b, avg_g, avg_r = [np.mean(img[:, :, i]) for i in range(3)]
        avg_gray = (avg_b + avg_g + avg_r) / 3
        scale = [avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r]
        for i in range(3):
            img[:, :, i] *= scale[i]
        return np.clip(img, 0, 255).astype(np.uint8)

    def _max_rgb_white_balance(self, image):
        img = image.astype(np.float32)
        max_b, max_g, max_r = [np.max(img[:, :, i]) for i in range(3)]
        max_val = max(max_b, max_g, max_r)
        scale = [max_val / max_b, max_val / max_g, max_val / max_r]
        for i in range(3):
            img[:, :, i] *= scale[i]
        return np.clip(img, 0, 255).astype(np.uint8)

    def _dcp_dehaze(self, img):
        omega = 0.95         # Weight for haze removal strength (higher → more haze removed; typical range: 0.85–0.98)
        t0 = 0.9             # Minimum transmission value to prevent over-brightening (typical range: 0.05–0.2)
        blockSize = 15       # Window size for dark channel computation (odd number; larger values → stronger smoothing)
        gimfiltR = 80        # Radius of guided filter (higher → smoother transmission map; try 20–80)
        eps = 1e-3           # Regularization for guided filter (lower → more detail preserved; try 1e-4 to 1e-2)
        percent = 0.001      # Top percentage of brightest dark pixels used for atmospheric light estimation (0.001–0.01)

        def get_min_channel(img):
            return np.min(img, axis=2)

        def get_dark_channel(img_min, block_size):
            return cv2.erode(img_min, np.ones((block_size, block_size), np.uint8))

        def get_atmospheric_light(dark, img, percent=0.001):
            h, w = dark.shape
            flat_dark = dark.ravel()
            flat_img = img.reshape(-1, 3)
            num_pixels = h * w
            num_top = max(int(percent * num_pixels), 1)
            indices = np.argpartition(flat_dark, -num_top)[-num_top:]
            brightest = flat_img[indices]
            return np.max(brightest, axis=0)

        def get_transmission(img_min, A, omega):
            return 1 - omega * img_min / A

        # Step 1: Estimate atmospheric light
        img_min = get_min_channel(img)
        dark = get_dark_channel(img_min, blockSize)
        A = get_atmospheric_light(dark, img, percent)

        # Step 2: Estimate transmission
        raw_trans = get_transmission(img_min.astype(np.float64), np.max(A), omega)

        # Step 3: Refine transmission using guided filter
        guided_filter = GuidedFilter(img, gimfiltR, eps)
        refined_trans = guided_filter.filter(raw_trans)
        refined_trans = np.clip(refined_trans, t0, 0.9)

        # Step 4: Recover the scene radiance
        img_float = img.astype(np.float64)
        result = np.empty_like(img_float)
        for i in range(3):
            result[:, :, i] = (img_float[:, :, i] - A[i]) / refined_trans + A[i]

        return np.clip(result, 0, 255).astype(np.uint8)


    def _udcp_dehaze(self, img):
        blockSize = self.blockSize

        # Step 1: Compute GB Dark Channel
        gb_dark = getDarkChannel(img, blockSize)

        # Step 2: Estimate Atmospheric Light
        A = getAtomsphericLight(gb_dark, img)

        # Step 3: Estimate Transmission Map (before refinement)
        raw_trans = getTransmission(img, A, blockSize)

        # Step 4: Refine Transmission using guided filter
        refined_trans = Refinedtransmission(raw_trans, img)

        # Step 5: Recover final image using transmission and A
        scene = sceneRadianceRGB(img, refined_trans, A)

        return np.clip(scene, 0, 255).astype(np.uint8)
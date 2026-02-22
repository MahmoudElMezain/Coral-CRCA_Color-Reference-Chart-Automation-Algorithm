import cv2
import numpy as np

class ImageEnhance:
    def __init__(self, algorithm_name="CLAHE", **kwargs):
        self.algorithm_name = algorithm_name.upper()

        if self.algorithm_name == "CLAHE":
            self.clip_limit = kwargs.get("clip_limit", 2.0)
            self.tile_grid_size = kwargs.get("tile_grid_size", (8, 8))

        elif self.algorithm_name == "GC":
            self.gamma = kwargs.get("gamma", 2.0)

        elif self.algorithm_name == "RS":
            self.alpha = kwargs.get("alpha", 0.5)
            self.blend_weight = kwargs.get("blend_weight", 0.5)


    def apply(self, image):
        """
        Applies the selected enhancement algorithm to the input image.

        Args:
            image (np.ndarray): Input image in BGR format

        Returns:
            np.ndarray: Enhanced image in BGR format
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy ndarray (e.g., from cv2.imread).")

        if self.algorithm_name == "CLAHE":
            return self._apply_clahe(image)
        elif self.algorithm_name == "GC":
            return self._apply_gamma_correction(image)
        elif self.algorithm_name == "RS":
            return self._apply_rayleigh_split_stretch(image)
        else:
            raise ValueError(f"Unsupported algorithm name: {self.algorithm_name}")

    def _apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _apply_gamma_correction(self, image):
        inv_gamma = 1.0 / self.gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _apply_rayleigh_split_stretch(self, image):
        """
        Applies Rayleigh Stretching based on histogram splitting.

        Returns:
            np.ndarray: Enhanced image in BGR format
        """
        img1 = image.copy().astype(np.float32)
        img2 = image.copy().astype(np.float32)

        b1, g1, r1 = cv2.split(img1)
        b2, g2, r2 = cv2.split(img2)

        imdb = ((b2.max() - b2.min()) / 2.0) + b2.min()
        imdg = ((g2.max() - g2.min()) / 2.0) + g2.min()
        imdr = ((r2.max() - r2.min()) / 2.0) + r2.min()

        alpha = self.alpha  # Rayleigh sigma

        # B
        b = b2.copy()
        b[b < imdb] = imdb
        b = (255.0 * (b - imdb)) / ((b2.max() - b2.min()) / (alpha ** 2))
        b = np.clip(b, 0, 255)

        # G
        g = g2.copy()
        g[g < imdg] = imdg
        g = (255.0 * (g - imdg)) / ((g2.max() - g2.min()) / (alpha ** 2))
        g = np.clip(g, 0, 255)

        # R
        r = r2.copy()
        r[r < imdr] = imdr
        r = (255.0 * (r - imdr)) / ((r2.max() - r2.min()) / (alpha ** 2))
        r = np.clip(r, 0, 255)

        res = cv2.merge((b, g, r)).astype(np.uint8)
        res1 = cv2.merge((b1, g1, r1)).astype(np.uint8)

        # Blending
        weight = self.blend_weight
        return cv2.addWeighted(res, weight, res1, 1 - weight, 0)

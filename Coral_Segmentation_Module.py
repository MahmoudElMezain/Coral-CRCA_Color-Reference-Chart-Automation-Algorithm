import os
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class CoralSCOPMaskGenerator:
    def __init__(self, model_type='vit_b', checkpoint_path='./checkpoints/vit_b_coralscop.pth',
                 iou_threshold=0.72, sta_threshold=0.62, point_number=32, gpu=0):
        """
        Initialize CoralSCOP SAM-based mask generator.

        Args:
            model_type (str): Model type ('vit_b').
            checkpoint_path (str): Path to SAM checkpoint.
            iou_threshold (float): IOU threshold for masks.
            sta_threshold (float): Stability threshold for masks.
            point_number (int): Number of sampling points per side.
            gpu (int): GPU device id to use.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = 'cuda'

        # Load SAM model
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)

        # Create mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=point_number,
            pred_iou_thresh=iou_threshold,
            stability_score_thresh=sta_threshold,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

    def generate_mask(self, image, largest_only=False):
        """
        Generate a mask for coral regions from a single image.

        Args:
            image (np.ndarray): Input BGR image.
            largest_only (bool): If True, return only the largest mask. If False, combine all masks.

        Returns:
            np.ndarray: Binary mask (uint8) with coral regions marked as 255.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid NumPy image (BGR).")

        # Convert image to RGB (SAM expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Generate masks
        masks = self.mask_generator.generate(image_rgb)

        if not masks:
            print("Warning: No masks generated for this image.")
            return np.zeros((height, width), dtype=np.uint8)

        if largest_only:
            # Find the mask with the largest area
            largest_mask = max(masks, key=lambda x: x['area'])
            seg = largest_mask['segmentation']
            final_mask = seg.astype(np.uint8) * 255
        else:
            # Combine all masks into one
            final_mask = np.zeros((height, width), dtype=np.uint8)
            for mask_info in masks:
                seg = mask_info['segmentation']
                binary_mask = seg.astype(np.uint8)
                final_mask = np.logical_or(final_mask, binary_mask).astype(np.uint8)
            final_mask = final_mask * 255  # Convert to 0/255

        return final_mask

    def apply_mask(self, image, largest_only=False):
        """
        Apply the generated mask to the original image.

        Args:
            image (np.ndarray): Input BGR image.
            largest_only (bool): If True, use only the largest mask.

        Returns:
            np.ndarray: Image with background masked out.
        """
        mask = self.generate_mask(image, largest_only=largest_only)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

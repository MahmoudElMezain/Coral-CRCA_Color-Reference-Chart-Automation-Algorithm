import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ultralytics.utils.ops import xyxyxyxy2xywhr

from Watch_Quadrant_Separator_Module import WatchQuadrantSeparatorModule
from Dominant_Color_Module import DominantColorModule
from Bleaching_Percentage_ModuleV3 import BleachingPercentageModule
from Color_Correction_Module import ColorCorrection
from Image_Enhancement_Module import ImageEnhance
from Coral_Segmentation_Module import CoralSCOPMaskGenerator


LINE_THICKNESS = 8
FONT_SCALE = 5
FONT_THICKNESS = 5


def show_bgr(title, image_bgr, scale=1.0):
    image = image_bgr
    if scale != 1.0:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.show(block=True)
    plt.close()


def show_pair_bgr(title_left, image_left, title_right, image_right, scale=1.0, legend_patches=None):
    left = image_left
    right = image_right
    if scale != 1.0:
        left = cv2.resize(left, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    axs[0].imshow(left_rgb)
    axs[0].set_title(title_left, fontsize=16)
    axs[0].axis("off")

    axs[1].imshow(right_rgb)
    axs[1].set_title(title_right, fontsize=16)
    axs[1].axis("off")

    if legend_patches:
        legend = axs[1].legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=10,
            frameon=True,
            borderpad=0.8,
            handlelength=1.5,
            handleheight=1.0,
            labelspacing=0.6
        )
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.show(block=True)
    plt.close()


def color_for_label(label):
    rng = np.random.RandomState(abs(hash(label)) % (2**32))
    return tuple(int(x) for x in rng.randint(0, 255, 3))


def detect_quadrants_from_image(separator, image_bgr):
    results = separator.model(image_bgr, verbose=False)[0]
    detections = {}

    for i in range(len(results.obb.xyxyxyxy)):
        conf = results.obb.conf[i].item()
        if conf < separator.conf_threshold:
            continue

        cls = int(results.obb.cls[i].item())
        label = separator.class_names.get(cls)
        if label not in separator.target_classes:
            continue

        xyxyxyxy = results.obb.xyxyxyxy[i][:8].cpu().numpy().flatten()
        xywhr = xyxyxyxy2xywhr(xyxyxyxy.reshape(1, -1))[0]

        if label not in detections or conf > detections[label]["conf"]:
            detections[label] = {"conf": conf, "xywhr": xywhr.tolist()}

    final_output = {}
    for label in separator.target_classes:
        if label in detections:
            final_output[label] = detections[label]["xywhr"]
        else:
            final_output[label] = []

    return final_output


def draw_detections(image_bgr, detections):
    output = image_bgr.copy()
    for label, xywhr in detections.items():
        if not xywhr:
            continue

        x_center, y_center, width, height, angle_rad = xywhr
        angle_deg = math.degrees(angle_rad)
        rect = ((x_center, y_center), (width, height), angle_deg)
        box = cv2.boxPoints(rect).astype(np.int32)
        color = color_for_label(label)

        cv2.polylines(output, [box], True, color, LINE_THICKNESS)
        text_pos = (box[0][0], box[0][1] - 5)
        cv2.putText(output, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color,
                    FONT_THICKNESS, cv2.LINE_AA)

    return output


def filter_sam_masks(masks, iou_threshold=0.72, stability_threshold=0.62):
    if not masks:
        return []

    filtered = []
    for m in masks:
        pred_iou = m.get("predicted_iou")
        stability = m.get("stability_score")

        keep = True
        if pred_iou is not None and pred_iou < iou_threshold:
            keep = False
        if stability is not None and stability < stability_threshold:
            keep = False

        if keep:
            filtered.append(m)

    if not filtered:
        filtered = masks

    filtered.sort(key=lambda x: x.get("area", 0), reverse=True)
    return filtered


def draw_mask_indices(image_bgr, masks):
    output = image_bgr.copy()
    for idx, m in enumerate(masks, start=1):
        seg = m["segmentation"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = color_for_label(f"mask_{idx}")
        cv2.drawContours(output, contours, -1, color, LINE_THICKNESS)

        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(output, str(idx), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color,
                            FONT_THICKNESS, cv2.LINE_AA)

    return output


if __name__ == "__main__":
    # === CONFIGURATION ===
    image_path = "Test_Images/CM02-CT004 RCO B-30.JPG"

    # YOLO-OBB model (adjust if you want a custom trained path)
    model_path = "Weights/YOLOv11X.pt"
    conf_threshold = 0.2

    # Dominant color & bleaching parameters
    image_size = 2000
    image_grid_size = 1
    quadrant_size = 256
    quadrant_grid_size = 4
    distance_mode = "cielab_euclidean"  # rgb_euclidean, cielab_euclidean, ciede2000

    # Color correction & enhancement
    apply_correction = True
    correction_type = "WB_GRAY"  # WB_GRAY, WB_RGB_MAX, DCP, UDCP
    enhancement_type = "CLAHE"   # CLAHE, GC, RS
    correction_kwargs = {"blockSize": 5} if correction_type == "UDCP" else {}
    enhancement_params = {
        "CLAHE": {"clip_limit": 3.5, "tile_grid_size": (12, 12)},
        "GC": {"gamma": 2.2},
        "RS": {"alpha": 3.1, "blend_weight": 0.5}
    }

    # SAM (CoralSCOP) segmentation
    sam_checkpoint = os.path.join("checkpoints", "vit_b_coralscop.pth")
    sam_model_type = "vit_b"
    mask_iou_threshold = 0.72
    mask_stability_threshold = 0.62
    sam_points_per_side = 32
    sam_gpu = 0

    # Manual palette for visualization (BGR)
    manual_palette = {
        "B1": (232, 248, 247), "B2": (190, 247, 244), "B3": (133, 241, 235),
        "B4": (58, 208, 202),  "B5": (25, 159, 153),  "B6": (0, 121, 101),
        "C1": (234, 236, 246), "C2": (193, 203, 243), "C3": (138, 158, 235),
        "C4": (68, 97, 201),   "C5": (36, 61, 153),   "C6": (1, 21, 116),
        "D1": (224, 236, 246), "D2": (192, 220, 244), "D3": (137, 189, 235),
        "D4": (83, 150, 207),  "D5": (27, 94, 151),   "D6": (0, 59, 115),
        "E1": (225, 242, 245), "E2": (191, 233, 245), "E3": (136, 214, 237),
        "E4": (72, 175, 207),  "E5": (31, 128, 156),  "E6": (0, 89, 116)
    }

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        raise SystemExit(1)

    # === Load image ===
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("[ERROR] Failed to read image.")
        raise SystemExit(1)

    # === Step 1: Color correction + enhancement ===
    enhanced_image = original_image.copy()
    if apply_correction:
        corrected = ColorCorrection(correction_type, **correction_kwargs).apply(enhanced_image)
        enhanced_image = ImageEnhance(enhancement_type, **enhancement_params[enhancement_type]).apply(corrected)

    show_pair_bgr("Original Image", original_image, "Denoised / Enhanced Image", enhanced_image, scale=0.6)

    # === Step 2: YOLO-OBB detection + dominant color extraction ===
    separator = WatchQuadrantSeparatorModule(model_path, conf_threshold)
    detections = detect_quadrants_from_image(separator, enhanced_image)

    detection_vis = draw_detections(enhanced_image, detections)
    show_bgr("YOLO-OBB Detection Result", detection_vis, scale=0.6)

    color_module = DominantColorModule(
        algorithm_name="dbscan",
        grid_size=int(quadrant_size / quadrant_grid_size),
        clusters=7,
        resize_dim=quadrant_size,
        convert_to_hsv=False
    )
    dominant_colors = color_module.extract_colors(enhanced_image, detections)
    bgr_references = {
        label: data["bgr"]
        for label, data in dominant_colors.items()
        if data is not None and "bgr" in data
    }

    # === Step 3: Coral segmentation (SAM) and mask selection ===
    mask_generator = CoralSCOPMaskGenerator(
        model_type=sam_model_type,
        checkpoint_path=sam_checkpoint,
        iou_threshold=mask_iou_threshold,
        sta_threshold=mask_stability_threshold,
        point_number=sam_points_per_side,
        gpu=sam_gpu
    )
    raw_masks = mask_generator.mask_generator.generate(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    filtered_masks = filter_sam_masks(raw_masks, mask_iou_threshold, mask_stability_threshold)

    if not filtered_masks:
        print("[ERROR] No masks were generated.")
        raise SystemExit(1)

    segmentation_vis = draw_mask_indices(enhanced_image, filtered_masks)
    show_bgr("Segmentation Result (Choose Mask Index)", segmentation_vis, scale=0.6)

    # Ask user to select a mask index
    while True:
        try:
            selection = int(input(f"Select a mask index (1-{len(filtered_masks)}): ").strip())
            if 1 <= selection <= len(filtered_masks):
                break
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

    chosen_mask = filtered_masks[selection - 1]["segmentation"].astype(np.uint8) * 255
    masked_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=chosen_mask)

    # === Step 4: Bleaching / color matching ===
    bleaching_module = BleachingPercentageModule(
        grid_size=int(image_size / image_grid_size),
        resize_dim=image_size,
        verbose=True,
        distance_mode=distance_mode
    )
    bleaching_percentage, bleaching_mask, dom_class = bleaching_module.compute(masked_image, bgr_references)
    print(f"\nBleaching Percentage: {bleaching_percentage:.2f}%")

    color_map = {i: manual_palette.get(f"{dom_class}{i}", (0, 0, 0)) for i in range(1, 7)}
    color_mask = np.zeros((*bleaching_mask.shape, 3), dtype=np.uint8)
    for i, color in color_map.items():
        color_mask[bleaching_mask == i] = color

    legend_patches = [
        mpatches.Patch(facecolor=np.array(color_map[i][::-1]) / 255.0,
                       edgecolor="black", label=f"{dom_class}{i}")
        for i in range(1, 7)
    ]

    # Visualization of final color matching 
    show_pair_bgr(
        "Selected Coral Mask Applied",
        masked_image,
        "Color Matching / Bleaching Map",
        color_mask,
        scale=0.6,
        legend_patches=legend_patches
    )

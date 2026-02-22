import cv2
import numpy as np
from scipy.spatial import distance
from collections import defaultdict
from colour.difference import delta_E
from colour.models import RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_Lab

'''
Optimized re-grouping and vectorized distance- CIEDE2000, Euclidean_RGB, and Euclidean_CIELAB
'''


class BleachingPercentageModule:
    def __init__(self, grid_size=32, resize_dim=512, verbose=True, distance_mode='ciede2000'):
        """
        distance_mode options:
        - 'rgb_euclidean'     : Euclidean distance in RGB space
        - 'cielab_euclidean'  : Euclidean distance in CIELAB space
        - 'ciede2000'         : Delta E 2000 color difference in CIELAB space
        """
        self.grid_size = grid_size
        self.resize_dim = resize_dim
        self.verbose = verbose
        self.distance_mode = distance_mode.lower()
        self.colourspace = RGB_COLOURSPACES['sRGB']
        self.whitepoint = self.colourspace.whitepoint

    def bgr_to_lab(self, bgr):
        """Convert a BGR tuple to Lab using the sRGB colourspace."""
        rgb = np.array(bgr[::-1]) / 255.0
        xyz = RGB_to_XYZ(rgb, colourspace=self.colourspace)
        return XYZ_to_Lab(xyz, self.whitepoint)

    def compute(self, coral_image_masked, dominant_colors):
        h, w = coral_image_masked.shape[:2]
        coral_resized = cv2.resize(coral_image_masked, (self.resize_dim, self.resize_dim), interpolation=cv2.INTER_AREA)
        block_h = self.resize_dim // self.grid_size
        block_w = self.resize_dim // self.grid_size

        # Prepare class groups
        class_groups = {'B': [f'B{i}' for i in range(1, 7)],
                        'C': [f'C{i}' for i in range(1, 7)],
                        'D': [f'D{i}' for i in range(1, 7)],
                        'E': [f'E{i}' for i in range(1, 7)]}
        dominant_bgr = {k: v for k, v in dominant_colors.items() if v is not None}

        # === Precompute Lab for references if needed ===
        ref_lab_dict = {}
        if self.distance_mode in ["cielab_euclidean", "ciede2000"]:
            for k, v in dominant_bgr.items():
                ref_lab_dict[k] = self.bgr_to_lab(v)

        # Prepare reference arrays for vectorized distance calculations
        ref_labels = list(dominant_bgr.keys())
        if self.distance_mode == "rgb_euclidean":
            ref_array = np.array([dominant_bgr[k] for k in ref_labels], dtype=np.float32)
        elif self.distance_mode == "cielab_euclidean":
            ref_array = np.array([ref_lab_dict[k] for k in ref_labels], dtype=np.float32)
        else:
            ref_array = None  # ciede2000 will be computed per reference

        superpixel_class_matches = []
        class_match_counts = {cls: defaultdict(int) for cls in class_groups}
        bleaching_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # === Step 1: Assign each superpixel to the closest reference color ===
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1, y2 = i * block_h, (i + 1) * block_h
                x1, x2 = j * block_w, (j + 1) * block_w
                block = coral_resized[y1:y2, x1:x2]

                if np.count_nonzero(block):
                    mask = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                    mask = cv2.inRange(mask, 1, 255)
                    avg_bgr = tuple(map(int, cv2.mean(block, mask=mask)[:3]))

                    # Convert once if using Lab
                    if self.distance_mode in ["cielab_euclidean", "ciede2000"]:
                        avg_lab = self.bgr_to_lab(avg_bgr)

                    # Vectorized distance calculation
                    if self.distance_mode == "rgb_euclidean":
                        dists = np.linalg.norm(ref_array - np.array(avg_bgr, dtype=np.float32), axis=1)
                    elif self.distance_mode == "cielab_euclidean":
                        dists = np.linalg.norm(ref_array - avg_lab, axis=1)
                    else:  # ciede2000 (cannot vectorize delta_E)
                        dists = np.array([delta_E(avg_lab, lab, method='CIE 2000')
                                          for lab in ref_lab_dict.values()])

                    min_index = np.argmin(dists)
                    best_label = ref_labels[min_index]

                    if best_label:
                        cls_prefix = best_label[0]
                        tone = int(best_label[1])
                        class_match_counts[cls_prefix][tone] += 1
                        superpixel_class_matches.append((i, j, cls_prefix, tone, avg_bgr))

        # === Step 2: Select the dominant class ===
        selected_class = None
        max_score = -1
        if self.verbose:
            print("\n=== Quadrant-wise Match Counts ===")
        for cls in class_groups:
            total = 0
            for grade in range(3, 7): ##EDIT
                count = class_match_counts[cls][grade]
                total += count
                if self.verbose:
                    print(f"{cls}{grade}: {count} pixels")
            if self.verbose:
                print(f"→ {cls} Total: {total} pixels\n")
            if total > max_score:
                selected_class = cls
                max_score = total
        if self.verbose:
            print(f"→ Selected Class: {selected_class}")

        # Prepare references for refinement
        grade_counts = defaultdict(int)
        selected_refs = [dominant_colors.get(f"{selected_class}{i}", None) for i in range(1, 7)]

        # Precompute Lab for selected refs if needed
        selected_refs_lab = []
        if self.distance_mode in ["cielab_euclidean", "ciede2000"]:
            for ref in selected_refs:
                selected_refs_lab.append(None if ref is None else self.bgr_to_lab(ref))
        else:
            selected_refs_lab = selected_refs

        # Prepare reference array for second pass (vectorized for Euclidean modes)
        if self.distance_mode == "rgb_euclidean":
            refine_ref_array = np.array([r for r in selected_refs if r is not None], dtype=np.float32)
        elif self.distance_mode == "cielab_euclidean":
            refine_ref_array = np.array([r for r in selected_refs_lab if r is not None], dtype=np.float32)
        else:
            refine_ref_array = None  # ciede2000 cannot be vectorized

        # === Step 3: Refine assignments ===
        for i, j, cls, tone, avg_bgr in superpixel_class_matches:
            if cls == selected_class and tone in [3, 4, 5, 6]: ##EDIT
                best_grade = tone
            else:
                if self.distance_mode in ["cielab_euclidean", "ciede2000"]:
                    avg_lab = self.bgr_to_lab(avg_bgr)

                if self.distance_mode == 'rgb_euclidean':
                    dists = np.linalg.norm(refine_ref_array - np.array(avg_bgr, dtype=np.float32), axis=1)
                    best_index = np.argmin(dists)
                    best_grade = best_index + 1
                elif self.distance_mode == 'cielab_euclidean':
                    dists = np.linalg.norm(refine_ref_array - avg_lab, axis=1)
                    best_index = np.argmin(dists)
                    best_grade = best_index + 1
                else:  # ciede2000
                    min_dist = float('inf')
                    best_grade = 0
                    for idx, ref_lab in enumerate(selected_refs_lab):
                        if ref_lab is None:
                            continue
                        dist = delta_E(avg_lab, ref_lab, method='CIE 2000')
                        if dist < min_dist:
                            min_dist = dist
                            best_grade = idx + 1

            grade_counts[best_grade] += 1
            bleaching_mask[i, j] = best_grade

        # === Final bleaching stats ===
        bleached = sum(grade_counts[g] for g in [1, 2, 3]) #EDIT
        healthy = sum(grade_counts[g] for g in [4,5, 6]) #EDIT
        total = bleached + healthy
        bleaching_percentage = (100.0 * bleached / total) if total > 0 else 0

        if self.verbose:
            print(f"\n=== Grade Counts for Class {selected_class} ===")
            for g in range(1, 7):
                print(f"{selected_class}{g}: {grade_counts[g]} pixels")
            print(f"\n=== Final Bleaching Report ===")
            print(f"Bleached: {bleached} pixels")
            print(f"Healthy: {healthy} pixels")
            print(f"Bleaching Percentage: {bleaching_percentage:.2f}%")

        bleaching_mask_full = cv2.resize(bleaching_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return bleaching_percentage, bleaching_mask_full, selected_class

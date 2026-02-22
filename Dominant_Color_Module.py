import cv2
import numpy as np
import math
from sklearn.cluster import    KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, Birch, OPTICS, DBSCAN

class DominantColorModule:
    def __init__(self, algorithm_name="mean", grid_size=1,clusters=4, resize_dim=256, convert_to_hsv=True):
        """
        Initializes the DominantColorModule with customization options.

        Args:
            algorithm_name (str): Color extraction method (e.g., "mean", "kmeans").
            grid_size (int): Placeholder for future superpixel or grid-based averaging.
            convert_to_hsv (bool): Whether to convert the image to HSV color space.
        """
        self.algorithm_name = algorithm_name
        self.grid_size = grid_size
        self.convert_to_hsv = convert_to_hsv
        self.clusters = clusters
        self.resize_dim = resize_dim

    def rotate_and_crop(self, image, center, size, angle):
        """
        Rotates the image around a center point and extracts an axis-aligned rectangle.

        Args:
            image (np.ndarray): Input image (BGR or HSV).
            center (tuple): (x, y) center of the rectangle.
            size (tuple): (width, height) of the rectangle.
            angle (float): Rotation angle in degrees (positive = counter-clockwise).

        Returns:
            np.ndarray: Cropped image patch (rotated and axis-aligned).
        """
        # Step 1: Compute rotation matrix (rotate image around original center)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Step 2: Rotate the full image
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Step 3: Transform the center to its new location
        rotated_center = np.dot(M, np.array([center[0], center[1], 1]))
        x, y = map(int, rotated_center)

        # Step 4: Compute axis-aligned bounding box around the rotated center
        w, h = map(int, size)
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        # Clip to bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(rotated.shape[1], x2)
        y2 = min(rotated.shape[0], y2)

        return rotated[y1:y2, x1:x2]

    def generate_superpixels(self, image, mask=None):
        """
        Divides an image into grid_size x grid_size blocks (superpixels),
        and computes the average color of each block.

        Args:
            image (np.ndarray): Input image patch (already rotated & resized).
            mask (np.ndarray): Optional binary mask (same shape as image, single channel).

        Returns:
            List of tuples: Each tuple is the average color of one grid block.
        """
        h, w = image.shape[:2]
        n = self.grid_size

        if h % n != 0 or w % n != 0:
            raise ValueError(f"Image size ({w}x{h}) must be divisible by grid_size {n}.")

        superpixels = []
        block_h, block_w = h // n, w // n

        for i in range(n):
            for j in range(n):
                y1, y2 = i * block_h, (i + 1) * block_h
                x1, x2 = j * block_w, (j + 1) * block_w
                block = image[y1:y2, x1:x2]

                if mask is not None:
                    block_mask = mask[y1:y2, x1:x2]
                    if np.count_nonzero(block_mask) == 0:
                        avg_color = (0, 0, 0)
                    else:
                        avg_color = cv2.mean(block, mask=block_mask)[:3]
                else:
                    avg_color = cv2.mean(block)[:3]

                superpixels.append(tuple(map(int, avg_color)))

        return superpixels


    def extract_colors(self, image, quadrant_dict):
        """
        Extract dominant colors from image regions using rotated bounding boxes by:
        - Masking the region inside each rotated box
        - Croping the bounding box around it
        - Rotating that patch upright
        - Re-croping to remove black corners
        - Resizing to a square patch
        - Computing the dominant color from masked pixels

        Args:
            image (np.ndarray): Input image as a NumPy array (BGR format by default).
            quadrant_dict (dict): Dictionary of form:
                                { "QuadrantName": [x_center, y_center, width, height, rotation] }
            output_size (tuple): Final patch size, e.g., (128, 128)

        Returns:
            dict: Each key is a quadrant name; value is either:
                - None (if not detected)
                - A dict with:
                    - hsv or bgr: tuple of average color (3 channels)
                    - box: 4 corner points of the original rotated rectangle
                    - patch: upright, resized, square image patch
        """

        # Ensure the input is a NumPy image
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected image as a NumPy array (from cv2.imread).")

        # Convert to HSV if required
        if self.convert_to_hsv:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Prepare the results dictionary
        results = {}

        # Loop through each quadrant in the dictionary
        for label, values in quadrant_dict.items():
            if (not values) or (label in ["Square_In", "Square_Out"]):
                results[label] = None
                continue

            # Extract YOLO-OBB format: center coordinates, size, and rotation (in radians)
            x_center, y_center, width, height, angle_rad = values

            # Convert angle from radians to degrees as required by OpenCV
            angle_deg = math.degrees(angle_rad)
            rect = ((x_center, y_center), (width, height), angle_deg)

            # Get the 4 corner points of the rotated box
            box = cv2.boxPoints(rect).astype(np.int32)  # shape (4, 2)

            # Create a binary mask covering the rotated quadrant region
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [box], 255)

            # Crop the axis-aligned bounding rectangle that encloses the rotated quadrant
            x, y, w, h = cv2.boundingRect(box)

            # Check that width/height are reasonable (box not degenerate)
            if w <= 1 or h <= 1:
                print(f"[WARNING] Skipping {label}: invalid box size.")
                results[label] = None
                continue

            ### Pad the original image and mask before cropping to avoid boundary errors
            padding = 50
            padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)
            padded_mask = cv2.copyMakeBorder(mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

            # Shift the bounding box and center by padding offset
            x += padding
            y += padding
            x_center_padded = x_center + padding
            y_center_padded = y_center + padding

            # Crop from padded versions
            cropped_img = padded_image[y:y+h, x:x+w]
            cropped_mask = padded_mask[y:y+h, x:x+w]
            center_rel = (x_center_padded - x, y_center_padded - y)

            # Recalculate the center relative to the cropped image
            center_rel = (x_center - x, y_center - y)



            # Compute rotation matrix to upright the region
            M = cv2.getRotationMatrix2D(center_rel, angle_deg, 1.0)

            try:
                # Rotate the cropped image and its mask
                rotated_patch = cv2.warpAffine(cropped_img, M, (w, h))
                rotated_mask = cv2.warpAffine(cropped_mask, M, (w, h))
            except cv2.error as e:
                print(f"[ERROR] warpAffine failed for {label}: {e}")
                results[label] = None
                continue

            # Ensure non-empty result after rotation
            if rotated_patch.size == 0 or rotated_mask.size == 0:
                print(f"[WARNING] Skipping {label}: empty result after rotation.")
                results[label] = None
                continue

            # Crop the rotated region again around the remaining non-black area
            x2, y2, w2, h2 = cv2.boundingRect(rotated_mask)
            if w2 <= 1 or h2 <= 1:
                print(f"[WARNING] Skipping {label}: invalid crop after rotation.")
                results[label] = None
                continue

            # Final cropped and masked patch
            tight_patch = rotated_patch[y2:y2+h2, x2:x2+w2]
            tight_mask = rotated_mask[y2:y2+h2, x2:x2+w2]

            # Resize to square patch for standardization
            try:
                resized_patch = cv2.resize(tight_patch, (self.resize_dim, self.resize_dim), interpolation=cv2.INTER_AREA)
                resized_mask = cv2.resize(tight_mask, (self.resize_dim, self.resize_dim), interpolation=cv2.INTER_NEAREST)
            except cv2.error as e:
                print(f"[ERROR] Resize failed for {label}: {e}")
                results[label] = None
                continue
            
            # === Compute superpixels and filter black/masked ones ===
            superpixel_colors = self.generate_superpixels(resized_patch, resized_mask)
            filtered_colors = [c for c in superpixel_colors if sum(c) > 0]

            # === Handle case: no valid pixels to analyze ===
            if not filtered_colors:
                print("No valid pixels to analyze")
                mean_color = (0, 0, 0)  # Fallback if mask excludes entire patch

            ## === STATISTICAL METHODS ===
            else:
                # Compute the dominant color from the masked region
                if self.algorithm_name == "mean_cv":  
                    # Built-in OpenCV method using pixel-wise mask
                    mean_color = cv2.mean(resized_patch, mask=resized_mask)[:3]

                elif self.algorithm_name == "mean":
                    # Arithmetic mean of all filtered superpixel colors
                    mean_color = tuple(map(int, np.mean(filtered_colors, axis=0)))

                elif self.algorithm_name == "median":
                    # Median of each channel across all filtered superpixels
                    mean_color = tuple(map(int, np.median(filtered_colors, axis=0)))

                elif self.algorithm_name == "mode":
                    # Most frequent exact color triplet
                    from collections import Counter
                    color_counter = Counter(filtered_colors)
                    mean_color = color_counter.most_common(1)[0][0]  # Most frequent tuple

                elif self.algorithm_name == "mode_tolerant":
                    # Mode with color binning tolerance (e.g., ±5 → bin size = 10)
                    from collections import defaultdict
                    bin_size = 10  # Adjustable tolerance level
                    bin_counts = defaultdict(list)

                    # Group colors into bins by quantization
                    for color in filtered_colors:
                        bin_key = tuple((np.array(color) // bin_size).astype(int))
                        bin_counts[bin_key].append(color)

                    # Find the bin with the most entries
                    dominant_bin = max(bin_counts.items(), key=lambda item: len(item[1]))[1]

                    # Compute average color within the dominant bin
                    mean_color = tuple(map(int, np.mean(dominant_bin, axis=0)))


                ## === CLUSTERING METHODS ===
                elif self.algorithm_name in ["kmeans", "affinity", "meanshift", "spectral", "agglomerative", "birch", "optics", "dbscan"]:
                    # === Cluster-based dominant color from filtered superpixels ===
                    data = np.array(filtered_colors, dtype=np.float32)

                    try:
                        method = self.algorithm_name.lower()

                        if method == "kmeans":
                            # KMeans: tune n_clusters
                            model = KMeans(n_clusters=min(self.clusters, len(data)), n_init='auto')
                            labels = model.fit_predict(data)

                        elif method == "affinity":
                            try:
                                # Affinity: tune the damping coeff. and maximum iterations
                                model = AffinityPropagation(damping=0.93, max_iter=1500)
                                labels = model.fit_predict(data)

                                # Guard against convergence failure (no exemplars)
                                if len(set(labels)) <= 1:
                                    raise RuntimeError("Affinity propagation found no clusters")

                            except Exception as e:
                                raise RuntimeError(f"AffinityPropagation failed: {e}")

                        elif method == "meanshift":
                            # MeanShift: auto-determines cluster count
                            model = MeanShift()
                            labels = model.fit_predict(data)

                        elif method == "spectral":
                            # Spectral Clustering: needs n_clusters
                            model = SpectralClustering(
                                n_clusters=min(self.clusters, len(data) - 1),
                                affinity='nearest_neighbors',
                                n_neighbors=min(10, len(data) - 1),
                                assign_labels='kmeans',
                                eigen_solver='arpack',
                                eigen_tol=1e-5,
                                random_state=42
                            )
                            labels = model.fit_predict(data)

                        ## === NEW CLUSTERING METHODS ===
                        elif method == "agglomerative":
                            # Agglomerative Clustering: hierarchical, works well for small data
                            model = AgglomerativeClustering(
                                n_clusters=min(self.clusters, len(data)),
                                linkage="ward" #linkage{‘ward’, ‘complete’, ‘average’, ‘single’}
                            )
                            labels = model.fit_predict(data)

                        elif method == "birch":
                            # Birch: fast, good for large datasets or streaming
                            model = Birch(n_clusters=min(self.clusters, len(data)))
                            labels = model.fit_predict(data)

                        elif method == "optics":
                            # OPTICS: density-based, can handle variable density clusters
                            model = OPTICS(min_samples=2, cluster_method="dbscan")
                            labels = model.fit_predict(data)

                        elif method == "dbscan":
                            # DBSCAN: density-based, identifies outliers as noise
                            model = DBSCAN(eps=1.5, min_samples=2)
                            labels = model.fit_predict(data)

                            # Filter out noise labels (label = -1)
                            unique_labels = set(labels)
                            if -1 in unique_labels:
                                unique_labels.remove(-1)
                            if not unique_labels:
                                raise RuntimeError("DBSCAN found only noise (no valid clusters)")
                            labels = np.array([l if l != -1 else -999 for l in labels])  # keep structure

                        else:
                            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm_name}")

                        # === Post-processing: find the most common cluster label ===
                        unique_labels = [l for l in set(labels) if l >= 0]
                        if not unique_labels:
                            raise RuntimeError("Clustering returned only noise or invalid labels")

                        dominant_label = max(unique_labels, key=lambda lbl: np.sum(labels == lbl))
                        dominant_cluster_colors = data[labels == dominant_label]

                        # Compute the mean color of that dominant cluster
                        mean_color = tuple(map(int, np.mean(dominant_cluster_colors, axis=0)))
                    

                    ## FALLBACK CONDITION
                    except Exception as e:
                        print(f"[ERROR] Clustering '{self.algorithm_name}' failed for {label}: {e}")
                        try:
                            # === Fallback to KMeans ===
                            fallback_model = KMeans(n_clusters=min(8, len(data)), n_init='auto') #MODIFY THIS
                            fallback_labels = fallback_model.fit_predict(data)

                            # Find dominant cluster from fallback
                            fallback_label = np.bincount(fallback_labels).argmax()
                            dominant_cluster_colors = data[fallback_labels == fallback_label]
                            mean_color = tuple(map(int, np.mean(dominant_cluster_colors, axis=0)))
                            print(f"[INFO] Fallback to KMeans succeeded for {label}")
                        except Exception as fallback_e:
                            print(f"[ERROR] Fallback KMeans also failed for {label}: {fallback_e}")
                            mean_color = tuple(map(int, np.mean(filtered_colors, axis=0)))  # Final fallback (Average)

                else:
                    raise NotImplementedError(f"Algorithm '{self.algorithm_name}' not implemented.")


            # Save the result for this quadrant
            results[label] = {
                "hsv" if self.convert_to_hsv else "bgr": tuple(map(int, mean_color)),  # Estimated color
                "box": box.tolist(),        # Original rotated quadrant (for visualization)
                "patch": resized_patch      # Upright, cropped, and square patch
            }

        return results



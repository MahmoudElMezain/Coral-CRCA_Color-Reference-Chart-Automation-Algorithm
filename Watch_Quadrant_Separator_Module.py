# Import the necessary YOLO and utility functions from Ultralytics
from ultralytics import YOLO
from ultralytics.utils.ops import xyxyxyxy2xywhr  # Converts 8-point polygon to rotated box format
import os

# Define the class for detecting CoralWatch quadrants and squares
class WatchQuadrantSeparatorModule:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        Constructor for the WatchQuadrantSeparatorModule.

        Args:
            model_path (str): Path to the YOLOv11-OBB model (.pt file)
            conf_threshold (float): Minimum confidence to accept detections
        """
        # Load the pretrained YOLOv11-OBB model
        self.model = YOLO(model_path)

        # Store the confidence threshold as an attribute
        self.conf_threshold = conf_threshold

        # Mapping from class index to readable label names
        self.class_names = {
            0: "Square_In", 1: "Square_Out",
            2: "B1", 3: "B2", 4: "B3", 5: "B4", 6: "B5", 7: "B6",
            8: "C1", 9: "C2", 10: "C3", 11: "C4", 12: "C5", 13: "C6",
            14: "D1", 15: "D2", 16: "D3", 17: "D4", 18: "D5", 19: "D6",
            20: "E1", 21: "E2", 22: "E3", 23: "E4", 24: "E5", 25: "E6"
        }

        # Store the list of all valid target labels
        self.target_classes = set(self.class_names.values())

    def detect(self, image_path):
        """
        Detects quadrants and squares in the image using YOLOv11-OBB.

        Args:
            image_path (str): File path to the input image.

        Returns:
            dict: A dictionary where each key is a quadrant name (e.g., 'C1') and
                  the value is a list of 5 numbers: [x_center, y_center, width, height, rotation].
                  Only the highest-confidence detection per class is returned.
                  If a class is not detected, its value is [].
        """

        # Run the YOLO model on the image (returns a list, take the first result)
        results = self.model(image_path, verbose=False)[0]

        # Create an empty dictionary to hold final detections
        detections = {}

        # Loop through each detection result
        for i in range(len(results.obb.xyxyxyxy)):
            # Get the confidence score of this detection
            conf = results.obb.conf[i].item()

            # Skip detection if confidence is below threshold
            if conf < self.conf_threshold:
                continue

            # Get the predicted class index (e.g., 8 for 'C1')
            cls = int(results.obb.cls[i].item())

            # Convert the class index to a human-readable label (e.g., 'C1')
            label = self.class_names.get(cls)

            # Only process labels that are part of our target list
            if label in self.target_classes:
                # Extract the 8 coordinates of the rotated bounding box
                xyxyxyxy = results.obb.xyxyxyxy[i][:8].cpu().numpy().flatten()

                # Convert polygon to rotated rectangle [x_center, y_center, width, height, rotation]
                xywhr = xyxyxyxy2xywhr(xyxyxyxy.reshape(1, -1))[0]


                # Only store the detection if it's:
                # - the first time seeing this label
                # - OR if it has a higher confidence than the previous one
                if label not in detections or conf > detections[label]["conf"]:
                    detections[label] = {
                        "conf": conf,              # Save confidence for later comparison
                        "xywhr": xywhr.tolist()    # Correct way to convert NumPy to list
                    }

        # Prepare the final output with all target labels included
        # If a class was not detected, assign it an empty list []
        final_output = {}
        for label in self.target_classes:
            if label in detections:
                # Add the highest-confidence detection for this label
                final_output[label] = detections[label]["xywhr"]
            else:
                # No detection found for this label → mark as empty
                final_output[label] = []

        return final_output


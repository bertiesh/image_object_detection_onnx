import onnxruntime as ort
import numpy as np
import cv2
import os

coco_classes = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class DetectionModel:
    def __init__(self, model_path: str):
        """Initialize the ONNX detection model."""
        self.model_path = model_path
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        # self.output_name = [output.name for output in self.session.get_outputs()]
        self.output_name = ["output"]
        self.img_size = 416

    def preprocess(self, image_path: str):
        """Preprocess image: Load, resize, normalize, and convert to tensor format."""
        image = cv2.imread(image_path)
        original_shape = image.shape[:2]  # Keep original shape for post-processing

        # Resize image to match model input size
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize pixel values (0-1 range)
        image = image.astype(np.float32) / 255.0

        # Transpose to (C, H, W) format as ONNX expects
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension (1, C, H, W)
        image = np.expand_dims(image, axis=0)

        return image, original_shape

    def postprocess(self, raw_outputs, original_shape, min_perc_prob):
        """Post-process the raw ONNX model outputs."""
        boxes, scores, labels = raw_outputs

        # Convert outputs to numpy arrays
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)

        # Filter out low-confidence detections
        valid_indices = scores > (min_perc_prob / 100.0)
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        labels = labels[valid_indices]

        # Rescale bounding boxes back to original image size
        h_factor, w_factor = original_shape
        scale_x = w_factor / self.img_size
        scale_y = h_factor / self.img_size
        for i in range(len(boxes)):
            boxes[i][0] *= scale_x  # xmin
            boxes[i][1] *= scale_y  # ymin
            boxes[i][2] *= scale_x  # xmax
            boxes[i][3] *= scale_y  # ymax

        # Format output as a list of dictionaries
        results = []
        for i in range(len(boxes)):
            class_index = labels[i]
            class_name = (
                coco_classes[class_index]
                if class_index < len(coco_classes)
                else "unknown"
            )

            result = {
                "name": class_name,
                "percentage_probability": scores[i] * 100,
                "box_points": boxes[i].tolist(),
            }
            results.append(result)

        return results

    def predict(self, image_path: str, output_image_path: str, min_perc_prob: int):
        """Run inference and return detected objects."""
        image, original_shape = self.preprocess(image_path)

        # Run ONNX model inference
        raw_outputs = self.session.run(self.output_name, {self.input_name: image})

        # Process the model outputs
        results = self.postprocess(raw_outputs, original_shape, min_perc_prob)

        # Draw bounding boxes on the image
        self.draw_boxes(image_path, results, output_image_path)

        return results

    def draw_boxes(self, image_path, results, output_path):
        """Draw detected object bounding boxes on the image."""
        image = cv2.imread(image_path)
        for obj in results:
            x_min, y_min, x_max, y_max = obj["box_points"]
            label = obj["name"]
            confidence = obj["percentage_probability"]

            # Draw rectangle
            cv2.rectangle(
                image,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 255, 0),
                2,
            )

            # Put label
            text = f"{label}: {confidence:.2f}%"
            cv2.putText(
                image,
                text,
                (int(x_min), int(y_min) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Save output image
        cv2.imwrite(output_path, image)

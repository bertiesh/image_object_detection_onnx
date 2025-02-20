import numpy as np
import onnxruntime
import cv2
from typing import List, Dict, Tuple, Optional, Union
import torch
from PIL import Image, ImageDraw, ImageFont


class DetectionModel:
    def __init__(self, model_path: str):
        """Initialize the ONNX model for object detection.

        Args:
            model_path: Path to the ONNX model file
        """
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.classes = self._load_classes()

    def _load_classes(self) -> List[str]:
        """Load class names for object detection."""
        return [
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

    def _preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess image for model input using PyTorch-style processing.

        Args:
            image_path: Path to input image

        Returns:
            Tuple of preprocessed image and original image dimensions
        """
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        original_size = image.shape[:2]

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        height = self.input_shape[2] if len(self.input_shape) > 2 else 416
        width = self.input_shape[3] if len(self.input_shape) > 3 else 416
        resized = cv2.resize(image, (width, height))

        # Normalize and convert to float32
        input_img = resized.astype(np.float32) / 255.0

        # Convert to NCHW format
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        return input_img, original_size

    def _postprocess_retinanet(
        self,
        outputs: List[np.ndarray],
        original_size: Tuple[int, int],
        min_confidence: float,
    ) -> Tuple[List[Dict], np.ndarray]:
        """Post-process RetinaNet model outputs.

        Args:
            outputs: Model output [boxes (N,4), scores (N,), class_ids (N,)]
            original_size: Original image dimensions
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (detection results, boxes tensor for visualization)
        """
        image_height, image_width = original_size
        scale_x = image_width / 416.0
        scale_y = image_height / 416.0

        # Get outputs
        boxes = outputs[0]  # Shape: (N, 4)
        scores = outputs[1]  # Shape: (N,)
        class_ids = outputs[2]  # Shape: (N,)

        # Filter by confidence
        mask = scores > min_confidence / 100
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Convert boxes to visualization format
        boxes_tensor = torch.tensor(boxes)

        # Scale boxes to original image size
        boxes_tensor[:, [0, 2]] *= scale_x
        boxes_tensor[:, [1, 3]] *= scale_y

        results = []
        for i in range(len(boxes)):
            results.append(
                {
                    "name": (
                        self.classes[class_ids[i]]
                        if class_ids[i] < len(self.classes)
                        else f"class_{class_ids[i]}"
                    ),
                    "percentage_probability": float(scores[i] * 100),
                    "box_points": boxes[i].tolist(),
                }
            )

        return results, boxes_tensor

    def predict(
        self, model_type: str, image_path: str, output_path: str, min_confidence: float
    ) -> List[Dict]:
        # Run object detection on an image and save the output image.
        # Preprocess image
        input_img, original_size = self._preprocess_image(image_path)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_img})

        # Post-process
        if model_type in ["yolov3", "tiny-yolov3"]:
            results = self._postprocess_yolo(outputs, original_size, min_confidence)
            boxes_tensor = torch.tensor([r["box_points"] for r in results])
        else:  # retina-net
            results, boxes_tensor = self._postprocess_retinanet(
                outputs, original_size, min_confidence
            )

        # Draw results on image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)

        # Draw boxes and labels
        labels = [r["name"] for r in results]

        output_image = draw_bounding_boxes_and_labels(
            image=image_tensor,
            boxes=boxes_tensor,
            draw_boxes=True,
            labels=labels,
            label_color=(0, 255, 0),  # Green text
            box_color=(0, 255, 0),  # Green boxes
            width=2,
            font_size=12,
        )

        # Convert back to OpenCV format and save
        output_np = output_image.permute(1, 2, 0).numpy()
        output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_np)

        return results


def draw_bounding_boxes_and_labels(
    image: torch.Tensor,
    boxes: torch.Tensor,
    draw_boxes: bool,
    labels: Optional[List[str]] = None,
    label_color: Optional[
        Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]
    ] = None,
    box_color: Optional[
        Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]
    ] = None,
    width: int = 1,
    font_size: int = 10,
) -> torch.Tensor:
    """Draw bounding boxes and labels on an image using PIL."""
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")

    # Convert to PIL Image
    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)

    # Convert boxes to integer coordinates
    boxes = boxes.to(torch.int64).tolist()

    # Draw boxes and labels
    for i, bbox in enumerate(boxes):
        if draw_boxes:
            draw.rectangle(bbox, outline=box_color, width=width)

        if labels and labels[i]:
            # Draw label above the box
            draw.text(
                (bbox[0], bbox[1] - font_size - 4),
                labels[i],
                fill=label_color,
                font=ImageFont.load_default(),
            )

    # Convert back to tensor
    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)

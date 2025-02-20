import argparse
import csv
import os
from pprint import pprint
from onnx_helper2 import DetectionModel

# -----------------------------------
# Argument Parsing
# python cmd_interface.py --image_path input_images/cars.png --output_path output_images/random.png --model_path object_detection_retina.onnx --output_csv_path first_csv.csv
# -----------------------------------
parser = argparse.ArgumentParser(description="Run object detection on a single image.")
parser.add_argument(
    "--image_path", type=str, required=True, help="Path to the input image."
)
parser.add_argument(
    "--output_csv_path",
    type=str,
    required=True,
    help="Path to save output CSV results.",
)
parser.add_argument(
    "--output_path", type=str, required=True, help="Path to save output JSON results."
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to the ONNX model file."
)
parser.add_argument(
    "--model_type",
    type=str,
    choices=["yolov3", "tiny-yolov3", "retina-net"],
    default="retina-net",
    help="Model type to use (default: retina-net).",
)
parser.add_argument(
    "--min_conf",
    type=int,
    default=30,
    help="Minimum confidence threshold for detections (0-100).",
)
args = parser.parse_args()

# Set input/output directories
input_path = os.path.join(os.getcwd(), args.image_path)
output_path = os.path.join(os.getcwd(), args.output_path)

# Load ONNX model
model = DetectionModel(args.model_path)

# Run detection on a single image
detections = model.predict(args.model_type, input_path, output_path, args.min_conf)

# Print results and save to JSON
pprint(detections)
with open(args.output_csv_path, "w", newline="") as out:
    csv_writer = csv.writer(out)
    csv_writer.writerow(
        ["name", "percentage_probability", "xmin", "ymin", "xmax", "ymax"]
    )
    for item in detections:
        csv_writer.writerow(
            [
                item["name"],
                item["percentage_probability"],
                item["box_points"][0],  # xmin
                item["box_points"][1],  # ymin
                item["box_points"][2],  # xmax
                item["box_points"][3],  # ymax
            ]
        )

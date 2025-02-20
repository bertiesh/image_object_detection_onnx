# Object Detection Onnx Model

This project provides an object detection system using ONNX models (RetinaNet). The system includes:
- A **CLI tool** for running object detection on images.
- A **Flask-ML backend server** for deploying the model as an API.
- **ONNXRuntime** for efficient inference with exported ONNX models.

## Features
- **ONNXRuntime** for fast model inference.
- **Flask-ML server** for serving detection results via API.
- **CLI tool** for local inference.
- Outputs **csv and png results** with detected objects.

---

## Exporting the ONNX Model

### **Exporting RetinaNet from ImageObjectDetection**
I use the **[image_object_detection](https://github.com/Shreneken/image_object_detection) repository** to export the ONNX model. Follow instructions on README.md to get the model running. Configure [Rescue Box Desktop](https://github.com/UMass-Rescue/RescueBox-Desktop/releases) to work with the repo using flask_ml==0.2.5. 


Set a breakpoint at Line 27, right before the `def predict` in the `initialize` in `detection_model.py` by adding the following line: `import pdb; pdb.set_trace()`. Send a request to the backend again using the same inputs from the RescueBox Desktop application. The breakpoint will be triggered in the backend. Run the following python code to export the ONNX model(https://drive.google.com/drive/folders/0ADDnIwVBJ46iUk9PVA?dmr=1&ec=wgc-drive-hero-goto). 

```bash
import torch
torch.onnx.export(self.detector_model._ObjectDetection__model, torch.randn(1, 3, 416, 416), "object_detection_retina.onnx", export_params=True, opset_version=16, do_constant_folding=True, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
```

## Setting Up the Environment

### **Install Dependencies**

Ensure ONNXRuntime is installed:

```bash
pip install onnx onnxruntime
```

### Create virtual environment

I use `pipenv` .

```
pipenv shell
```

## Using the CLI Tool

### **Running Object Detection**

The CLI allows you to run object detection on a **single image**.

```bash
python cmd_interface.py --image_path cars.png --output_path random.png --model_path object_detection_retina.onnx --output_csv_path first_csv.csv
```

## Running the Flask-ML Server

### **Start the Server**

```bash
python model_server_onnx.py
```

## Project Requirements Checklist

- ✅ **ONNX Model Export** (image_object_detection)
- ✅ **ONNXRuntime for Inference**
- ✅ **CLI Tool**
- ✅ **Flask-ML Backend Server**
- ✅ **JSON Output for Results**
- ✅ **Formatted with `black`**

## Expected Output

When running object detection on an image, the project outputs:

- A csv file with detected objects.
- An image with bounding boxes.

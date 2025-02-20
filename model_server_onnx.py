from typing import TypedDict

from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    EnumParameterDescriptor,
    EnumVal,
    FileInput,
    FileResponse,
    FileType,
    InputSchema,
    InputType,
    IntRangeDescriptor,
    ParameterSchema,
    RangedIntParameterDescriptor,
    ResponseBody,
    TaskSchema,
)

from enum import Enum, auto
from onnx_helper2 import DetectionModel
import csv
import os


class Model_Server:

    server = MLServer(__name__)

    server.add_app_metadata(
        name="Image Object Detection",
        author="UMass Rescue",
        version="0.2.0",
        info=load_file_as_string("IOD_app.md"),
    )

    @staticmethod
    def create_object_detection_task_schema() -> TaskSchema:
        input_path_schema = InputSchema(
            key="input_path", label="Input Image", input_type=InputType.FILE
        )
        output_img_schema = InputSchema(
            key="output_img", label="Output Image", input_type=InputType.FILE
        )
        output_csv_schema = InputSchema(
            key="output_csv", label="Output CSV", input_type=InputType.FILE
        )

        min_perc_prob_schema = ParameterSchema(
            key="min_perc_prob",
            label="Minimum Percentage Probability",
            value=RangedIntParameterDescriptor(
                range=IntRangeDescriptor(min=0, max=100), default=30
            ),
        )
        model_type_schema = ParameterSchema(
            key="model_type",
            label="Model Type",
            value=EnumParameterDescriptor(
                enum_vals=[
                    EnumVal(label="Yolov3", key="yolov3"),
                    EnumVal(label="Tiny Yolov3", key="tiny-yolov3"),
                    EnumVal(label="Retina Net", key="retina-net"),
                ],
                default="retina-net",
            ),
        )

        return TaskSchema(
            inputs=[input_path_schema, output_img_schema, output_csv_schema],
            parameters=[min_perc_prob_schema, model_type_schema],
        )

    class DetectionInputs(TypedDict):
        input_path: FileInput
        output_img: FileInput
        output_csv: FileInput

    class DetectionParameters(TypedDict):
        min_perc_prob: int
        model_type: str
        min_perc_prob: int
        model_type: str
        model_type: str
        model_type: str

    class Model_Type(Enum):
        yolov3 = auto()
        tiny_yolov3 = auto()
        retina_net = auto()

    @staticmethod
    @server.route(
        "/detect",
        task_schema_func=create_object_detection_task_schema,
        short_title="Detect Objects",
        order=0,
    )
    def detect(inputs: DetectionInputs, parameters: DetectionParameters):
        exec_path = os.getcwd()
        input_path = os.path.join(exec_path, inputs["input_path"].path)
        output_img_path = os.path.join(exec_path, inputs["output_img"].path)
        output_csv_path = inputs["output_csv"].path
        min_perc_prob = parameters["min_perc_prob"]
        model_type = parameters["model_type"]
        results = []
        # Initialize appropriate model based on type
        if model_type == "retina-net":
            model = DetectionModel("object_detection_retina.onnx")
            results = model.predict(
                "retina-net", input_path, output_img_path, min_perc_prob
            )
        # elif model_type == "yolov3":
        #     model = DetectionModel("object_detection_yolov3.onnx")
        #     results = model.predict('yolov3', input_path, output_img_path, min_perc_prob)
        # elif model_type == "tiny-yolov3":
        #     model = DetectionModel("object_detection_tiny.onnx")
        #     results = model.predict('tiny-yolov3', input_path, output_img_path, min_perc_prob)

        with open(output_csv_path, "w", newline="") as out:
            csv_writer = csv.writer(out)
            csv_writer.writerow(
                ["name", "percentage_probability", "xmin", "ymin", "xmax", "ymax"]
            )
            for item in results:
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

        return ResponseBody(
            root=FileResponse(file_type=FileType.CSV, path=output_csv_path)
        )

    @classmethod
    def start_server(cls):
        cls.server.run()


model_server = Model_Server()

if __name__ == "__main__":
    model_server.start_server()

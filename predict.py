import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        with open(api_json_file, "r") as file:
            workflow = json.load(file)

        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(self, input_file: Path, filename: str):
        dest_path = os.path.join(INPUT_DIR, filename)
        shutil.copy(input_file, dest_path)
        return filename

    def update_workflow(self, workflow, **kwargs):
        garment_filename = kwargs.get("garment_image")
        model_filename = kwargs.get("model_image")

        print("Updating workflow...")
        for node_id, node in workflow.items():
            if node["class_type"] == "LoadImage":
                if node["inputs"]["image"] == "garment_image":
                    print(f"Setting garment image for node {node_id}")
                    node["inputs"]["image"] = garment_filename
                elif node["inputs"]["image"] == "model_image":
                    print(f"Setting model image for node {node_id}")
                    node["inputs"]["image"] = model_filename

    def predict(
        self,
        garment_image: Path = Input(
            description="Garment image file",
            default='https://nikhil-tayal.blr1.digitaloceanspaces.com/RE_0042_24-11-23GG.3330.jpg',
        ),
        model_image: Path = Input(
            description="Model image file",
            default='https://nikhil-tayal.blr1.digitaloceanspaces.com/RE_0042_24-11-23GG.3330.jpg',
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = Input(description="Seed for randomness", default=42),
    ) -> List[Path]:
        print("Starting prediction...")
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        garment_filename = self.filename_with_extension(garment_image, "garment")
        model_filename = self.filename_with_extension(model_image, "model")

        self.handle_input_file(garment_image, garment_filename)
        self.handle_input_file(model_image, model_filename)

        print("Garment image copied as:", garment_filename)
        print("Model image copied as:", model_filename)

        with open(api_json_file, "r") as file:
            workflow = json.load(file)

        self.update_workflow(
            workflow,
            garment_image=garment_filename,
            model_image=model_filename,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()

        print("Running workflow...")
        result = self.comfyUI.run_workflow(wf)

        print("Workflow result:", result)

        output_files = self.comfyUI.get_files(OUTPUT_DIR)
        print("Output files:", output_files)

        return optimise_images.optimise_image_files(
            output_format, output_quality, output_files
        )

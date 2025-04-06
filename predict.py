import os
import shutil
import json
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images

INPUT_DIR = "/tmp/inputs"
OUTPUT_DIR = "/tmp/outputs"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR]
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
api_json_file = "workflow_api.json"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Load workflow to prepare weights (none in this simple one)
        with open(api_json_file, "r") as f:
            workflow = json.load(f)
        self.comfyUI.handle_weights(workflow, weights_to_download=[])

    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
    ) -> List[Path]:

        # Clean up previous outputs
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Copy input image to INPUT_DIR
        input_filename = "input_image.png"
        shutil.copy(input_image, os.path.join(INPUT_DIR, input_filename))

        # Load workflow and update input
        with open(api_json_file, "r") as f:
            workflow = json.load(f)
        workflow["1"]["inputs"]["image"] = input_filename

        # Load & run workflow
        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        output_files = self.comfyUI.get_files([OUTPUT_DIR])
        if not output_files:
            raise RuntimeError("No output files generated.")

        return optimise_images.optimise_image_files(output_format, output_quality, output_files)

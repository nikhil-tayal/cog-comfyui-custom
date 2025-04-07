import json
import os
import base64
import tempfile
from typing import Dict
from PIL import Image

from cog import BasePredictor, Input, Path

# ComfyUI core imports (modify these if your repo structure is different)
import folder_paths
import execution
import nodes
import comfy.model_management

class Predictor(BasePredictor):
    def setup(self):
        # Load the workflow as a Python dict
        with open("workflow_api.json", "r") as f:
            self.workflow = json.load(f)

    def predict(
        self,
        garment_image: Path = Input(description="Garment image"),
        model_image: Path = Input(description="Model image"),
    ) -> Path:
        # Make a deep copy so we don't mutate the original
        workflow = json.loads(json.dumps(self.workflow))

        # Replace input image placeholders
        for node in workflow.values():
            if node["class_type"] == "LoadImage":
                if node["inputs"]["image"] == "garment_image":
                    node["inputs"]["image"] = str(garment_image)
                elif node["inputs"]["image"] == "model_image":
                    node["inputs"]["image"] = str(model_image)

        # Save modified workflow to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            json.dump(workflow, f)
            f.flush()
            workflow_path = f.name

        # Initialize model manager if not already
        comfy.model_management.set_torch_device()

        # Execute the workflow
        output_images = execution.process_workflow(workflow_path)

        # Assume one output image for simplicity
        if output_images:
            first_image = list(output_images.values())[0][0]
            return Path(first_image)
        else:
            raise RuntimeError("No output image was generated.")

"""
This file can be used to build a Truss for deployment with Baseten.
If used, it should be renamed to model.py and placed alongside the other
files from /riffusion in the standard /model directory of the Truss.

For more on the Truss file format, see https://truss.baseten.co/
"""

import typing as T

import dacite
import torch
from huggingface_hub import snapshot_download

from riffusion.datatypes import InferenceInput
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.server import compute_request


class Model:
    """
    Baseten Truss model class for riffusion.

    See: https://truss.baseten.co/reference/structure#model.py
    """

    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._pipeline = None
        self._vae = None

        self.checkpoint_name = "riffusion/riffusion-model-v1"

        # Download entire seed image folder from huggingface hub
        self._seed_images_dir = snapshot_download(self.checkpoint_name, allow_patterns="*.png")

    def load(self):
        """
        Load the model. Guaranteed to be called before `predict`.
        """
        self._pipeline = RiffusionPipeline.load_checkpoint(
            checkpoint=self.checkpoint_name,
            use_traced_unet=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def preprocess(self, request: T.Dict) -> T.Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def predict(self, request: T.Dict) -> T.Dict[str, T.List]:
        """
        This is the main function that is called.
        """
        assert self._pipeline is not None, "Model pipeline not loaded"

        try:
            inputs = dacite.from_dict(InferenceInput, request)
        except dacite.exceptions.WrongTypeError as exception:
            return str(exception), 400
        except dacite.exceptions.MissingValueError as exception:
            return str(exception), 400

        # NOTE: Autocast disabled to speed up inference, previous inference time was 10s on T4
        with torch.inference_mode() and torch.cuda.amp.autocast(enabled=False):
            response = compute_request(
                inputs=inputs,
                pipeline=self._pipeline,
                seed_images_dir=self._seed_images_dir,
            )

        return response

    def postprocess(self, request: T.Dict) -> T.Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

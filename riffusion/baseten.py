"""
This file can be used to build a Truss for deployment with Baseten.
If used, it should be renamed to model.py and placed alongside the other
files from /riffusion in the standard /model directory of the Truss.

For more on the Truss file format, see https://truss.baseten.co/
"""

import base64
import dataclasses
import json
import io
from pathlib import Path
from typing import Dict, List

import PIL
import torch
import dacite

from huggingface_hub import hf_hub_download, snapshot_download

from .audio import wav_bytes_from_spectrogram_image, mp3_bytes_from_wav_bytes
from .datatypes import InferenceInput, InferenceOutput
from .riffusion_pipeline import RiffusionPipeline


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._model = None
        self._vae = None

        # Download entire seed image folder from huggingface hub
        self._seed_images_dir = snapshot_download(
            "riffusion/riffusion-model-v1", allow_patterns="*.png"
        )

    def load(self):
        # Load Riffusion model here and assign to self._model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available() == False:
            # Use only if you don't have a GPU with fp16 support
            self._model = RiffusionPipeline.from_pretrained(
                "riffusion/riffusion-model-v1",
                safety_checker=lambda images, **kwargs: (images, False),
            ).to(device)
        else:
            # Model loading the model with fp16. This will fail if ran without a GPU with fp16 support
            pipe = RiffusionPipeline.from_pretrained(
                "riffusion/riffusion-model-v1",
                revision="fp16",
                torch_dtype=torch.float16,
                # Disable the NSFW filter, causes incorrect false positives
                safety_checker=lambda images, **kwargs: (images, False),
            ).to(device)

            # Deliberately not implementing channels_Last as it resulted in slower inference pipeline
            # pipe.unet.to(memory_format=torch.channels_last)

            @dataclasses.dataclass
            class UNet2DConditionOutput:
                sample: torch.FloatTensor

            # Use traced unet from hf hub
            unet_file = hf_hub_download(
                "riffusion/riffusion-model-v1", filename="unet_traced.pt", subfolder="unet_traced"
            )
            unet_traced = torch.jit.load(unet_file)

            class TracedUNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.in_channels = pipe.unet.in_channels
                    self.device = pipe.unet.device

                def forward(self, latent_model_input, t, encoder_hidden_states):
                    sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                    return UNet2DConditionOutput(sample=sample)

            pipe.unet = TracedUNet()

            self._model = pipe

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        """
        This is the main function that is called.
        """
        # Example request:
        # {"alpha":0.25,"num_inference_steps":50,"seed_image_id":"og_beat","mask_image_id":None,"start":{"prompt":"lo-fi beat for the holidays","seed":906295,"denoising":0.75,"guidance":7},"end":{"prompt":"lo-fi beat for the holidays","seed":906296,"denoising":0.75,"guidance":7}}

        # Parse an InferenceInput dataclass from the payload
        try:
            inputs = dacite.from_dict(InferenceInput, request)
        except dacite.exceptions.WrongTypeError as exception:
            # logging.info(json_data)
            return str(exception), 400
        except dacite.exceptions.MissingValueError as exception:
            # logging.info(json_data)
            return str(exception), 400

        # NOTE: Autocast disabled to speed up inference, previous inference time was 10s on T4
        with torch.inference_mode() and torch.cuda.amp.autocast(enabled=False):
            response = self.compute(inputs)

        return response

    def compute(self, inputs: InferenceInput) -> str:
        """
        Does all the heavy lifting of the request.
        """
        # Load the seed image by ID
        init_image_path = Path(self._seed_images_dir, f"seed_images/{inputs.seed_image_id}.png")

        if not init_image_path.is_file():
            return f"Invalid seed image: {inputs.seed_image_id}", 400
        init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

        # Load the mask image by ID
        if inputs.mask_image_id:
            mask_image_path = Path(self._seed_images_dir, f"seed_images/{inputs.mask_image_id}.png")
            if not mask_image_path.is_file():
                return f"Invalid mask image: {inputs.mask_image_id}", 400
            mask_image = PIL.Image.open(str(mask_image_path)).convert("RGB")
        else:
            mask_image = None

        # Execute the model to get the spectrogram image
        image = self._model.riffuse(inputs, init_image=init_image, mask_image=mask_image)

        # Reconstruct audio from the image
        wav_bytes, duration_s = wav_bytes_from_spectrogram_image(image)
        mp3_bytes = mp3_bytes_from_wav_bytes(wav_bytes)

        # Compute the output as base64 encoded strings
        image_bytes = self.image_bytes_from_image(image, mode="JPEG")

        # Assemble the output dataclass
        output = InferenceOutput(
            image="data:image/jpeg;base64," + self.base64_encode(image_bytes),
            audio="data:audio/mpeg;base64," + self.base64_encode(mp3_bytes),
            duration_s=duration_s,
        )

        return json.dumps(dataclasses.asdict(output))

    def image_bytes_from_image(self, image: PIL.Image, mode: str = "PNG") -> io.BytesIO:
        """
        Convert a PIL image into bytes of the given image format.
        """
        image_bytes = io.BytesIO()
        image.save(image_bytes, mode)
        image_bytes.seek(0)
        return image_bytes

    def base64_encode(self, buffer: io.BytesIO) -> str:
        """
        Encode the given buffer as base64.
        """
        return base64.encodebytes(buffer.getvalue()).decode("ascii")

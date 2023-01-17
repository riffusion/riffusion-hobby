"""
Prediction interface for Cog ⚙️
https://github.com/replicate/cog/blob/main/docs/python.md
"""

import argparse
import os
import shutil
import typing as T

import numpy as np
import PIL
import torch
from cog import BaseModel, BasePredictor, Input, Path

from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

MODEL_ID = "riffusion/riffusion-model-v1"
MODEL_CACHE = "riffusion-cache"

# Where built-in seed images are stored
SEED_IMAGES_DIR = Path("./seed_images")
SEED_IMAGES = [val.split(".")[0] for val in os.listdir(SEED_IMAGES_DIR) if "png" in val]
SEED_IMAGES.sort()


class Output(BaseModel):
    """
    Output class for riffusion predictions
    """

    audio: Path
    spectrogram: Path
    error: T.Optional[str] = None


class RiffusionPredictor(BasePredictor):
    """
    Implementation of cog predictor object s.t. we can run riffusion predictions w/cog.

    See README & https://github.com/replicate/cog for details
    """

    def setup(self, local_files_only=True):
        """
        Loads the model onto GPU from local cache.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = RiffusionPipeline.load_checkpoint(
            checkpoint=MODEL_ID,
            use_traced_unet=True,
            device=self.device,
            local_files_only=local_files_only,
            cache_dir=MODEL_CACHE,
        )

    def predict(
        self,
        prompt_a: str = Input(description="The prompt for your audio", default="funky synth solo"),
        denoising: float = Input(
            description="How much to transform input spectrogram",
            default=0.75,
            ge=0,
            le=1,
        ),
        prompt_b: str = Input(
            description="The second prompt to interpolate with the first,"
            "leave blank if no interpolation",
            default=None,
        ),
        alpha: float = Input(
            description="Interpolation alpha if using two prompts."
            "A value of 0 uses prompt_a fully, a value of 1 uses prompt_b fully",
            default=0.5,
            ge=0,
            le=1,
        ),
        num_inference_steps: int = Input(
            description="Number of steps to run the diffusion model", default=50, ge=1
        ),
        seed_image_id: str = Input(
            description="Seed spectrogram to use", default="vibes", choices=SEED_IMAGES
        ),
    ) -> Output:
        """
        Runs riffusion inference.
        """
        # Load the seed image by ID
        init_image_path = Path(SEED_IMAGES_DIR, f"{seed_image_id}.png")
        if not init_image_path.is_file():
            return Output(error=f"Invalid seed image: {seed_image_id}")
        init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

        # fake max ints
        seed_a = np.random.randint(0, 2147483647)
        seed_b = np.random.randint(0, 2147483647)

        start = PromptInput(prompt=prompt_a, seed=seed_a, denoising=denoising)
        if not prompt_b:  # no transition
            prompt_b = prompt_a
            alpha = 0
        end = PromptInput(prompt=prompt_b, seed=seed_b, denoising=denoising)
        riffusion_input = InferenceInput(
            start=start,
            end=end,
            alpha=alpha,
            num_inference_steps=num_inference_steps,
            seed_image_id=seed_image_id,
        )

        # Execute the model to get the spectrogram image
        image = self.model.riffuse(riffusion_input, init_image=init_image, mask_image=None)

        # Reconstruct audio from the image
        params = SpectrogramParams()
        converter = SpectrogramImageConverter(params=params, device=self.device)
        segment = converter.audio_from_spectrogram_image(image)

        if not os.path.exists("out/"):
            os.mkdir("out")

        out_img_path = "out/spectrogram.jpg"
        image.save("out/spectrogram.jpg", exif=image.getexif())

        out_wav_path = "out/gen_sound.wav"
        segment.export(out_wav_path, format="wav")

        return Output(audio=Path(out_wav_path), spectrogram=Path(out_img_path))


# TODO(hayk): Can we get rid of the below functions and incorporate into
# RiffusionPipeline.load_checkpoint?


def download_weights():
    """
    Clears local cache & downloads riffusion weights
    """
    if os.path.exists(MODEL_CACHE):
        shutil.rmtree(MODEL_CACHE)
    os.makedirs(MODEL_CACHE)

    pred = RiffusionPredictor()
    pred.setup(local_files_only=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_weights", action="store_true", help="Download and cache weights"
    )
    args = parser.parse_args()
    if args.download_weights:
        download_weights()

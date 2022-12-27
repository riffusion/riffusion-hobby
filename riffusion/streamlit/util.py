"""
Streamlit utilities (mostly cached wrappers around riffusion code).
"""
import io
import typing as T

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

# TODO(hayk): Add URL params


@st.experimental_singleton
def load_riffusion_checkpoint(
    checkpoint: str = "riffusion/riffusion-model-v1",
    no_traced_unet: bool = False,
    device: str = "cuda",
) -> RiffusionPipeline:
    """
    Load the riffusion pipeline.
    """
    return RiffusionPipeline.load_checkpoint(
        checkpoint=checkpoint,
        use_traced_unet=not no_traced_unet,
        device=device,
    )


@st.experimental_singleton
def load_stable_diffusion_pipeline(
    checkpoint: str = "riffusion/riffusion-model-v1",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """
    Load the riffusion pipeline.

    TODO(hayk): Merge this into RiffusionPipeline to just load one model.
    """
    if device == "cpu" or device.lower().startswith("mps"):
        print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
        dtype = torch.float32

    return StableDiffusionPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=dtype,
        safety_checker=lambda images, **kwargs: (images, False),
    ).to(device)


@st.experimental_memo
def run_txt2img(
    prompt: str,
    num_inference_steps: int,
    guidance: float,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    device: str = "cuda",
) -> Image.Image:
    """
    Run the text to image pipeline with caching.
    """
    pipeline = load_stable_diffusion_pipeline(device=device)

    generator_device = "cpu" if device.lower().startswith("mps") else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    output = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        negative_prompt=negative_prompt or None,
        generator=generator,
        width=width,
        height=height,
    )

    return output["images"][0]


@st.experimental_singleton
def spectrogram_image_converter(
    params: SpectrogramParams,
    device: str = "cuda",
) -> SpectrogramImageConverter:
    return SpectrogramImageConverter(params=params, device=device)


@st.experimental_memo
def audio_bytes_from_spectrogram_image(
    image: Image.Image,
    params: SpectrogramParams,
    device: str = "cuda",
    output_format: str = "mp3",
) -> io.BytesIO:
    converter = spectrogram_image_converter(params=params, device=device)
    segment = converter.audio_from_spectrogram_image(image)

    audio_bytes = io.BytesIO()
    segment.export(audio_bytes, format=output_format)
    audio_bytes.seek(0)

    return audio_bytes


def select_device(container: T.Any = st.sidebar) -> str:
    """
    Dropdown to select a torch device, with an intelligent default.
    """
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"

    device_options = ["cuda", "cpu", "mps"]
    device = st.sidebar.selectbox(
        "Device", options=device_options, index=device_options.index(default_device)
    )
    assert device is not None

    return device

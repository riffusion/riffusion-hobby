"""
Streamlit utilities (mostly cached wrappers around riffusion code).
"""
import io
import typing as T

import pydub
import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

from riffusion.audio_splitter import AudioSplitter
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


@st.experimental_singleton
def load_stable_diffusion_img2img_pipeline(
    checkpoint: str = "riffusion/riffusion-model-v1",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionImg2ImgPipeline:
    """
    Load the image to image pipeline.

    TODO(hayk): Merge this into RiffusionPipeline to just load one model.
    """
    if device == "cpu" or device.lower().startswith("mps"):
        print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
        dtype = torch.float32

    return StableDiffusionImg2ImgPipeline.from_pretrained(
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


@st.cache
def spectrogram_image_from_audio(
    segment: pydub.AudioSegment,
    params: SpectrogramParams,
    device: str = "cuda",
) -> Image.Image:
    converter = spectrogram_image_converter(params=params, device=device)
    return converter.spectrogram_image_from_audio(segment)


@st.experimental_memo
def audio_segment_from_spectrogram_image(
    image: Image.Image,
    params: SpectrogramParams,
    device: str = "cuda",
) -> pydub.AudioSegment:
    converter = spectrogram_image_converter(params=params, device=device)
    return converter.audio_from_spectrogram_image(image)


@st.experimental_memo
def audio_bytes_from_spectrogram_image(
    image: Image.Image,
    params: SpectrogramParams,
    device: str = "cuda",
    output_format: str = "mp3",
) -> io.BytesIO:
    segment = audio_segment_from_spectrogram_image(image=image, params=params, device=device)

    audio_bytes = io.BytesIO()
    segment.export(audio_bytes, format=output_format)

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
        "Device",
        options=device_options,
        index=device_options.index(default_device),
        help="Which compute device to use. CUDA is recommended.",
    )
    assert device is not None

    return device


@st.experimental_memo
def load_audio_file(audio_file: io.BytesIO) -> pydub.AudioSegment:
    return pydub.AudioSegment.from_file(audio_file)


@st.experimental_singleton
def get_audio_splitter(device: str = "cuda"):
    return AudioSplitter(device=device)


@st.cache
def run_img2img(
    prompt: str,
    init_image: Image.Image,
    denoising_strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    seed: int,
    device: str = "cuda",
    progress_callback: T.Optional[T.Callable[[float], T.Any]] = None,
) -> Image.Image:
    pipeline = load_stable_diffusion_img2img_pipeline(device=device)

    generator_device = "cpu" if device.lower().startswith("mps") else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    num_expected_steps = max(int(num_inference_steps * denoising_strength), 1)

    def callback(step: int, tensor: torch.Tensor, foo: T.Any) -> None:
        if progress_callback is not None:
            progress_callback(step / num_expected_steps)

    result = pipeline(
        prompt=prompt,
        image=init_image,
        strength=denoising_strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt or None,
        num_images_per_prompt=1,
        generator=generator,
        callback=callback,
        callback_steps=1,
    )

    return result.images[0]

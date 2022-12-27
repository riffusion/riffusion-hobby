import io
from pathlib import Path

import dacite
from diffusers import StableDiffusionPipeline
import streamlit as st
import torch
from PIL import Image

from riffusion.datatypes import InferenceInput
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


@st.experimental_singleton
def load_stable_diffusion_pipeline(
    checkpoint: str = "riffusion/riffusion-model-v1",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """
    Load the riffusion pipeline.
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

    generator = torch.Generator(device="cpu").manual_seed(seed)

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


def render_text_to_audio() -> None:
    """
    Render audio from text.
    """
    prompt = st.text_input("Prompt")
    if not prompt:
        st.info("Enter a prompt")
        return

    negative_prompt = st.text_input("Negative prompt")
    seed = st.sidebar.number_input("Seed", value=42)
    num_inference_steps = st.sidebar.number_input("Inference steps", value=20)
    width = st.sidebar.number_input("Width", value=512)
    height = st.sidebar.number_input("Height", value=512)
    guidance = st.sidebar.number_input(
        "Guidance", value=7.0, help="How much the model listens to the text prompt"
    )

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

    image = run_txt2img(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance=guidance,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        device=device,
    )

    st.image(image)

    # TODO(hayk): Change the frequency range to [20, 20k] once the model is retrained
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
    )

    segment = streamlit_util.audio_from_spectrogram_image(
        image=image,
        params=params,
        device=device,
    )

    mp3_bytes = io.BytesIO()
    segment.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)
    st.audio(mp3_bytes)


if __name__ == "__main__":
    render_text_to_audio()

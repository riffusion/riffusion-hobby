import io
from pathlib import Path

import dacite
import streamlit as st
import torch
from PIL import Image

from riffusion.datatypes import InferenceInput
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


def render_interpolation_demo() -> None:
    """
    Render audio from text.
    """
    prompt = st.text_input("Prompt", label_visibility="collapsed")
    if not prompt:
        st.info("Enter a prompt")
        return

    seed = st.sidebar.number_input("Seed", value=42)
    denoising = st.sidebar.number_input("Denoising", value=0.01)
    guidance = st.sidebar.number_input("Guidance", value=7.0)
    num_inference_steps = st.sidebar.number_input("Inference steps", value=50)

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

    pipeline = streamlit_util.load_riffusion_checkpoint(device=device)

    input_dict = {
        "alpha": 0.75,
        "num_inference_steps": num_inference_steps,
        "seed_image_id": "og_beat",
        "start": {
            "prompt": prompt,
            "seed": seed,
            "denoising": denoising,
            "guidance": guidance,
        },
        "end": {
            "prompt": prompt,
            "seed": seed,
            "denoising": denoising,
            "guidance": guidance,
        },
    }
    st.json(input_dict)

    inputs = dacite.from_dict(InferenceInput, input_dict)

    # TODO fix
    init_image_path = Path(__file__).parent.parent.parent.parent / "seed_images" / "og_beat.png"
    init_image = Image.open(str(init_image_path)).convert("RGB")

    # Execute the model to get the spectrogram image
    image = pipeline.riffuse(
        inputs,
        init_image=init_image,
        mask_image=None,
    )
    st.image(image)

    # TODO(hayk): Change the frequency range to [20, 20k] once the model is retrained
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
    )

    # Reconstruct audio from the image
    # TODO(hayk): It may help performance to cache this object
    converter = SpectrogramImageConverter(params=params, device=str(pipeline.device))
    segment = converter.audio_from_spectrogram_image(
        image,
        apply_filters=True,
    )

    mp3_bytes = io.BytesIO()
    segment.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)
    st.audio(mp3_bytes)


if __name__ == "__main__":
    render_interpolation_demo()

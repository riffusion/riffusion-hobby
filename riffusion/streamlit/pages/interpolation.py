import dataclasses
import io
import typing as T
from pathlib import Path

import numpy as np
import pydub
import streamlit as st
from PIL import Image

from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


def render_interpolation_demo() -> None:
    st.set_page_config(layout="wide", page_icon="🎸")

    st.subheader(":performing_arts: Interpolation")
    st.write(
        """
    Interpolate between prompts in the latent space.
    """
    )

    # Sidebar params

    device = streamlit_util.select_device(st.sidebar)

    num_interpolation_steps = T.cast(
        int,
        st.sidebar.number_input(
            "Interpolation steps",
            value=4,
            min_value=1,
            max_value=20,
            help="Number of model generations between the two prompts. Controls the duration.",
        ),
    )

    num_inference_steps = T.cast(
        int,
        st.sidebar.number_input(
            "Steps per sample", value=50, help="Number of denoising steps per model run"
        ),
    )

    init_image_name = st.sidebar.selectbox(
        "Seed image",
        # TODO(hayk): Read from directory
        options=["og_beat", "agile", "marim", "motorway", "vibes", "custom"],
        index=0,
        help="Which seed image to use for img2img",
    )
    assert init_image_name is not None
    if init_image_name == "custom":
        init_image_file = st.sidebar.file_uploader(
            "Upload a custom seed image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if init_image_file:
            st.sidebar.image(init_image_file)

    show_individual_outputs = st.sidebar.checkbox(
        "Show individual outputs",
        value=False,
        help="Show each model output",
    )
    show_images = st.sidebar.checkbox(
        "Show individual images",
        value=False,
        help="Show each generated image",
    )

    # Prompt inputs A and B in two columns

    with st.form(key="interpolation_form"):
        left, right = st.columns(2)

        with left:
            st.write("##### Prompt A")
            prompt_input_a = get_prompt_inputs(key="a")

        with right:
            st.write("##### Prompt B")
            prompt_input_b = get_prompt_inputs(key="b")

        st.form_submit_button("Generate", type="primary")

    if not prompt_input_a.prompt or not prompt_input_b.prompt:
        st.info("Enter both prompts to interpolate between them")
        return

    # TODO(hayk): Make not linspace
    alphas = list(np.linspace(0, 1, num_interpolation_steps))
    alphas_str = ", ".join([f"{alpha:.2f}" for alpha in alphas])
    st.write(f"**Alphas** : [{alphas_str}]")

    if init_image_name == "custom":
        if not init_image_file:
            st.info("Upload a custom seed image")
            return
        init_image = Image.open(init_image_file).convert("RGB")
    else:
        init_image_path = (
            Path(__file__).parent.parent.parent.parent / "seed_images" / f"{init_image_name}.png"
        )
        init_image = Image.open(str(init_image_path)).convert("RGB")

    # TODO(hayk): Move this code into a shared place and add to riffusion.cli
    image_list: T.List[Image.Image] = []
    audio_bytes_list: T.List[io.BytesIO] = []
    for i, alpha in enumerate(alphas):
        inputs = InferenceInput(
            alpha=float(alpha),
            num_inference_steps=num_inference_steps,
            seed_image_id="og_beat",
            start=prompt_input_a,
            end=prompt_input_b,
        )

        if i == 0:
            with st.expander("Example input JSON", expanded=False):
                st.json(dataclasses.asdict(inputs))

        image, audio_bytes = run_interpolation(
            inputs=inputs,
            init_image=init_image,
            device=device,
        )

        if show_individual_outputs:
            st.write(f"#### ({i + 1} / {len(alphas)}) Alpha={alpha:.2f}")
            if show_images:
                st.image(image)
            st.audio(audio_bytes)

        image_list.append(image)
        audio_bytes_list.append(audio_bytes)

    st.write("#### Final Output")

    # TODO(hayk): Concatenate with better blending
    audio_segments = [pydub.AudioSegment.from_file(audio_bytes) for audio_bytes in audio_bytes_list]
    concat_segment = audio_segments[0]
    for segment in audio_segments[1:]:
        concat_segment = concat_segment.append(segment, crossfade=0)

    audio_bytes = io.BytesIO()
    concat_segment.export(audio_bytes, format="mp3")
    audio_bytes.seek(0)

    st.write(f"Duration: {concat_segment.duration_seconds:.3f} seconds")
    st.audio(audio_bytes)


def get_prompt_inputs(key: str) -> PromptInput:
    """
    Compute prompt inputs from widgets.
    """
    prompt = st.text_input("Prompt", label_visibility="collapsed", key=f"prompt_{key}")
    seed = T.cast(int, st.number_input("Seed", value=42, key=f"seed_{key}"))
    denoising = st.number_input(
        "Denoising", value=0.75, key=f"denoising_{key}", help="How much to modify the seed image"
    )
    guidance = st.number_input(
        "Guidance",
        value=7.0,
        key=f"guidance_{key}",
        help="How much the model listens to the text prompt",
    )

    return PromptInput(
        prompt=prompt,
        seed=seed,
        denoising=denoising,
        guidance=guidance,
    )


@st.experimental_memo
def run_interpolation(
    inputs: InferenceInput, init_image: Image.Image, device: str = "cuda"
) -> T.Tuple[Image.Image, io.BytesIO]:
    """
    Cached function for riffusion interpolation.
    """
    pipeline = streamlit_util.load_riffusion_checkpoint(
        device=device,
        # No trace so we can have variable width
        no_traced_unet=True,
    )

    image = pipeline.riffuse(
        inputs,
        init_image=init_image,
        mask_image=None,
    )

    # TODO(hayk): Change the frequency range to [20, 20k] once the model is retrained
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
    )

    # Reconstruct from image to audio
    audio_bytes = streamlit_util.audio_bytes_from_spectrogram_image(
        image=image,
        params=params,
        device=device,
        output_format="mp3",
    )

    return image, audio_bytes


if __name__ == "__main__":
    render_interpolation_demo()

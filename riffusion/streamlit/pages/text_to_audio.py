import typing as T

import streamlit as st

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


def render_text_to_audio() -> None:
    """
    Render audio from text.
    """
    prompt = st.text_input("Prompt")
    negative_prompt = st.text_input("Negative prompt")
    seed = T.cast(int, st.sidebar.number_input("Seed", value=42))
    num_inference_steps = T.cast(int, st.sidebar.number_input("Inference steps", value=50))
    width = T.cast(int, st.sidebar.number_input("Width", value=512))
    height = T.cast(int, st.sidebar.number_input("Height", value=512))
    guidance = st.sidebar.number_input(
        "Guidance", value=7.0, help="How much the model listens to the text prompt"
    )

    if not prompt:
        st.info("Enter a prompt")
        return

    device = streamlit_util.select_device(st.sidebar)

    image = streamlit_util.run_txt2img(
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

    audio_bytes = streamlit_util.audio_bytes_from_spectrogram_image(
        image=image,
        params=params,
        device=device,
        output_format="mp3",
    )
    st.audio(audio_bytes)


if __name__ == "__main__":
    render_text_to_audio()

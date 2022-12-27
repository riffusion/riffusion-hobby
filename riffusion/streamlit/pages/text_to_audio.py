import typing as T

import streamlit as st

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


def render_text_to_audio() -> None:
    st.set_page_config(layout="wide", page_icon="🎸")

    st.subheader(":pencil2: Text to Audio")
    st.write(
        """
    Generate audio from text prompts.  \nRuns the model directly without a seed image or
    interpolation.
    """
    )

    device = streamlit_util.select_device(st.sidebar)

    prompt = st.text_input("Prompt")
    negative_prompt = st.text_input("Negative prompt")

    with st.sidebar.expander("Text to Audio Params", expanded=True):
        seed = T.cast(int, st.number_input("Seed", value=42))
        num_inference_steps = T.cast(int, st.number_input("Inference steps", value=50))
        width = T.cast(int, st.number_input("Width", value=512))
        guidance = st.number_input(
            "Guidance", value=7.0, help="How much the model listens to the text prompt"
        )

    if not prompt:
        st.info("Enter a prompt")
        return

    image = streamlit_util.run_txt2img(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance=guidance,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=512,
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

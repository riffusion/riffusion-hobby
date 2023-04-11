import typing as T

import streamlit as st

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


def render() -> None:
    st.subheader("🌊 Text to Audio")
    st.write(
        """
    Generate audio from text prompts.
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            This tool runs riffusion in the simplest text to image form to generate an audio
            clip from a text prompt. There is no seed image or interpolation here. This mode
            allows more diversity and creativity than when using a seed image, but it also
            leads to having less control. Play with the seed to get infinite variations.
            """
        )

    device = streamlit_util.select_device(st.sidebar)
    extension = streamlit_util.select_audio_extension(st.sidebar)
    checkpoint = streamlit_util.select_checkpoint(st.sidebar)

    with st.form("Inputs"):
        prompt = st.text_input("Prompt")
        negative_prompt = st.text_input("Negative prompt")

        row = st.columns(4)
        num_clips = T.cast(
            int,
            row[0].number_input(
                "Number of clips",
                value=1,
                min_value=1,
                max_value=25,
                help="How many outputs to generate (seed gets incremented)",
            ),
        )
        starting_seed = T.cast(
            int,
            row[1].number_input(
                "Seed",
                value=42,
                help="Change this to generate different variations",
            ),
        )

        st.form_submit_button("Riff", type="primary")

    with st.sidebar:
        num_inference_steps = T.cast(int, st.number_input("Inference steps", value=30))
        width = T.cast(int, st.number_input("Width", value=512))
        guidance = st.number_input(
            "Guidance", value=7.0, help="How much the model listens to the text prompt"
        )
        scheduler = st.selectbox(
            "Scheduler",
            options=streamlit_util.SCHEDULER_OPTIONS,
            index=0,
            help="Which diffusion scheduler to use",
        )
        assert scheduler is not None

        use_20k = st.checkbox("Use 20kHz", value=False)

    if not prompt:
        st.info("Enter a prompt")
        return

    if use_20k:
        params = SpectrogramParams(
            min_frequency=10,
            max_frequency=20000,
            sample_rate=44100,
            stereo=True,
        )
    else:
        params = SpectrogramParams(
            min_frequency=0,
            max_frequency=10000,
            stereo=False,
        )

    seed = starting_seed
    for i in range(1, num_clips + 1):
        st.write(f"#### Riff {i} / {num_clips} - Seed {seed}")

        image = streamlit_util.run_txt2img(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=512,
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )
        st.image(image)

        segment = streamlit_util.audio_segment_from_spectrogram_image(
            image=image,
            params=params,
            device=device,
        )

        streamlit_util.display_and_download_audio(
            segment, name=f"{prompt.replace(' ', '_')}_{seed}", extension=extension
        )

        seed += 1

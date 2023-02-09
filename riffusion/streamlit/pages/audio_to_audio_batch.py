import io
import typing as T

import numpy as np
import pydub
import streamlit as st
from PIL import Image

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util
from random import randint


def render_audio_to_audio_batch() -> None:
    st.set_page_config(layout="wide", page_icon="🎸")

    st.subheader(":wave: Audio to Audio Batch")
    st.write(
        """
    Generate audio in batches from audio file and text prompt.
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            This tool allows you to upload an audio file of arbitrary length and modify it with
            a text prompt. It does this by slicing the source audio to to a spectrogram, then 
            performing an img2img style transfer with riffusion in randomly seeded batches.

            Try a denoising strength of 0.4 for light modification and 0.55 for more heavy
            modification. The best specific denoising depends on how different the prompt is
            from the source audio. You can play with the seed to get infinite variations.
            """
        )

    device = streamlit_util.select_device(st.sidebar)

    audio_file = st.file_uploader(
        "Upload audio",
        type=["mp3", "m4a", "ogg", "wav", "flac"],
        label_visibility="collapsed",
    )

    if not audio_file:
        st.info("Upload audio to get started")
        return

    st.write("#### Original")
    st.audio(audio_file)

    segment = streamlit_util.load_audio_file(audio_file)

    segment = segment.set_frame_rate(44100)
    st.write(f"Duration: {segment.duration_seconds:.2f}s, Sample Rate: {segment.frame_rate}Hz")

    if "counter" not in st.session_state:
        st.session_state.counter = 0

    def increment_counter():
        st.session_state.counter += 1

    cols = st.columns(3)
    clip_start_time_s = cols[0].number_input(
        "Start Time [s]",
        min_value=0.0,
        value=0.0,
    )
    clip_duration_s = cols[1].number_input(
        "Clip Duration [s]",
        min_value=3.0,
        max_value=10.0,
        value=5.0,
    )
    batches = int(cols[2].number_input(
        "Batches",
        min_value=1,
        max_value=100,
        value=1,
    ))
    st.write(
        f"Slicing clip of duration {clip_duration_s}s "
        f"over {batches} batches."
    )

    clip_start_time_ms = int(clip_start_time_s * 1000)
    clip_duration_ms = int(clip_duration_s * 1000)
    clip_segment: pydub.AudioSegment = segment[clip_start_time_ms: clip_start_time_ms + clip_duration_ms]

    audio_bytes = io.BytesIO()
    clip_segment.export(audio_bytes, format="wav")

    params = SpectrogramParams()

    init_image: Image.Image = streamlit_util.spectrogram_image_from_audio(
        clip_segment,
        params=params,
        device=device,
    )

    closest_width = int(np.ceil(init_image.width / 32) * 32)
    closest_height = int(np.ceil(init_image.height / 32) * 32)
    init_image_resized = init_image.resize((closest_width, closest_height), Image.BICUBIC)

    st.write("#### Source Clip")
    with st.expander("Source Clip"):
        st.audio(audio_bytes)
        st.image(init_image, use_column_width=False)

    st.write("#### Settings")
    with st.form("Conversion Params"):

        prompt = st.text_input("Text Prompt")
        negative_prompt = st.text_input("Negative Prompt")

        cols = st.columns(3)
        denoising_strength = cols[0].number_input(
            "Denoising Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
        )
        guidance_scale = cols[1].number_input(
            "Guidance Scale",
            min_value=0.0,
            max_value=20.0,
            value=7.0,
        )
        num_inference_steps = int(
            cols[2].number_input(
                "Num Inference Steps",
                min_value=1,
                max_value=150,
                value=50,
            )
        )

        submit_button = st.form_submit_button("Convert", on_click=increment_counter)

    if not prompt:
        st.info("Enter a prompt")
        return

    if not submit_button:
        return

    for b in range(0, batches):
        # Each batch has a random seed.
        seed = randint(10, 100000)

        container = st.container()
        empty_bin = container.empty()
        progress_callback = None

        container.write(f"### Seed {seed}")

        with empty_bin.container():
            st.info(f"Riffing ({b}/{batches})...")
            progress = st.progress(0.0)
            progress_callback = progress.progress

        left, right = container.columns(2)

        left.write(f"##### Riffed Clip")

        result_image: Image.Image = streamlit_util.run_img2img(
            prompt=prompt,
            init_image=init_image_resized,
            denoising_strength=denoising_strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            progress_callback=progress_callback,
            device=device,
        )

        # Resize back to original size
        result_image = result_image.resize(init_image.size, Image.BICUBIC)


        left.image(result_image, use_column_width=False)

        riffed_segment: pydub.AudioSegment = streamlit_util.audio_segment_from_spectrogram_image(
            image=result_image,
            params=params,
            device=device,
        )

        audio_bytes = io.BytesIO()
        riffed_segment.export(audio_bytes, format="wav")
        left.audio(audio_bytes)

        right.write(f"##### Differential")
        diff_np = np.maximum(
            0, np.asarray(init_image).astype(np.float32) - np.asarray(result_image).astype(np.float32)
        )
        diff_image: Image.Image = Image.fromarray(255 - diff_np.astype(np.uint8))
        diff_segment = streamlit_util.audio_segment_from_spectrogram_image(
            image=diff_image,
            params=params,
            device=device,
        )
        right.image(diff_image, use_column_width=False)

        audio_bytes = io.BytesIO()
        diff_segment.export(audio_bytes, format="wav")
        right.audio(audio_bytes)

        empty_bin.empty()


@st.cache
def test(segment: pydub.AudioSegment, counter: int) -> int:
    st.write("#### Trimmed")
    st.write(segment.duration_seconds)
    return counter


if __name__ == "__main__":
    render_audio_to_audio_batch()

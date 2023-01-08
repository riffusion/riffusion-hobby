import io
import typing as T

import numpy as np
import pydub
import streamlit as st
from PIL import Image

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


def render_audio_to_audio() -> None:
    st.set_page_config(layout="wide", page_icon="🎸")

    st.subheader(":wave: Audio to Audio")
    st.write(
        """
    Modify existing audio from a text prompt.
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            This tool allows you to upload an audio file of arbitrary length and modify it with
            a text prompt. It does this by sweeping over the audio in overlapping clips, doing
            img2img style transfer with riffusion, then stitching the clips back together with
            cross fading to eliminate seams.

            Try a denoising strength of 0.4 for light modification and 0.55 for more heavy
            modification. The best specific denoising depends on how different the prompt is
            from the source audio. You can play with the seed to get infinite variations.
            Currently the same seed is used for all clips along the track.
            """
        )

    device = streamlit_util.select_device(st.sidebar)

    audio_file = st.file_uploader(
        "Upload audio",
        type=["mp3", "m4a", "ogg", "wav", "flac", "webm"],
        label_visibility="collapsed",
    )

    if not audio_file:
        st.info("Upload audio to get started")
        return

    st.write("#### Original")
    st.audio(audio_file)

    segment = streamlit_util.load_audio_file(audio_file)

    # TODO(hayk): Fix
    segment = segment.set_frame_rate(44100)
    st.write(f"Duration: {segment.duration_seconds:.2f}s, Sample Rate: {segment.frame_rate}Hz")

    if "counter" not in st.session_state:
        st.session_state.counter = 0

    def increment_counter():
        st.session_state.counter += 1

    cols = st.columns(4)
    start_time_s = cols[0].number_input(
        "Start Time [s]",
        min_value=0.0,
        value=0.0,
    )
    duration_s = cols[1].number_input(
        "Duration [s]",
        min_value=0.0,
        value=15.0,
    )
    clip_duration_s = cols[2].number_input(
        "Clip Duration [s]",
        min_value=3.0,
        max_value=10.0,
        value=5.0,
    )
    overlap_duration_s = cols[3].number_input(
        "Overlap Duration [s]",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
    )

    duration_s = min(duration_s, segment.duration_seconds - start_time_s)
    increment_s = clip_duration_s - overlap_duration_s
    clip_start_times = start_time_s + np.arange(0, duration_s - clip_duration_s, increment_s)
    st.write(
        f"Slicing {len(clip_start_times)} clips of duration {clip_duration_s}s "
        f"with overlap {overlap_duration_s}s."
    )

    with st.expander("Clip Times"):
        st.dataframe(
            {
                "Start Time [s]": clip_start_times,
                "End Time [s]": clip_start_times + clip_duration_s,
                "Duration [s]": clip_duration_s,
            }
        )

    with st.form("Conversion Params"):

        prompt = st.text_input("Text Prompt")
        negative_prompt = st.text_input("Negative Prompt")

        cols = st.columns(4)
        denoising_strength = cols[0].number_input(
            "Denoising Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
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

        seed = int(
            cols[3].number_input(
                "Seed",
                min_value=0,
                value=42,
            )
        )

        submit_button = st.form_submit_button("Convert", on_click=increment_counter)

    # TODO fix

    show_clip_details = st.sidebar.checkbox("Show Clip Details", True)
    show_difference = st.sidebar.checkbox("Show Difference", False)

    clip_segments: T.List[pydub.AudioSegment] = []
    for i, clip_start_time_s in enumerate(clip_start_times):
        clip_start_time_ms = int(clip_start_time_s * 1000)
        clip_duration_ms = int(clip_duration_s * 1000)
        clip_segment = segment[clip_start_time_ms : clip_start_time_ms + clip_duration_ms]

        # TODO(hayk): I don't think this is working properly
        if i == len(clip_start_times) - 1:
            silence_ms = clip_duration_ms - int(clip_segment.duration_seconds * 1000)
            if silence_ms > 0:
                clip_segment = clip_segment.append(pydub.AudioSegment.silent(duration=silence_ms))

        clip_segments.append(clip_segment)

    if not prompt:
        st.info("Enter a prompt")
        return

    if not submit_button:
        return

    params = SpectrogramParams()

    result_images: T.List[Image.Image] = []
    result_segments: T.List[pydub.AudioSegment] = []
    for i, clip_segment in enumerate(clip_segments):
        st.write(f"### Clip {i} at {clip_start_times[i]}s")

        audio_bytes = io.BytesIO()
        clip_segment.export(audio_bytes, format="wav")

        init_image = streamlit_util.spectrogram_image_from_audio(
            clip_segment,
            params=params,
            device=device,
        )

        # TODO(hayk): Roll this into spectrogram_image_from_audio?
        # TODO(hayk): Scale something when computing audio
        closest_width = int(np.ceil(init_image.width / 32) * 32)
        closest_height = int(np.ceil(init_image.height / 32) * 32)
        init_image_resized = init_image.resize((closest_width, closest_height), Image.BICUBIC)

        progress_callback = None
        if show_clip_details:
            left, right = st.columns(2)

            left.write("##### Source Clip")
            left.image(init_image, use_column_width=False)
            left.audio(audio_bytes)

            right.write("##### Riffed Clip")
            empty_bin = right.empty()
            with empty_bin.container():
                st.info("Riffing...")
                progress = st.progress(0.0)
                progress_callback = progress.progress

        image = streamlit_util.run_img2img(
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
        image = image.resize(init_image.size, Image.BICUBIC)

        result_images.append(image)

        if show_clip_details:
            empty_bin.empty()
            right.image(image, use_column_width=False)

        riffed_segment = streamlit_util.audio_segment_from_spectrogram_image(
            image=image,
            params=params,
            device=device,
        )
        result_segments.append(riffed_segment)

        audio_bytes = io.BytesIO()
        riffed_segment.export(audio_bytes, format="wav")

        if show_clip_details:
            right.audio(audio_bytes)

        if show_clip_details and show_difference:
            diff_np = np.maximum(
                0, np.asarray(init_image).astype(np.float32) - np.asarray(image).astype(np.float32)
            )
            diff_image = Image.fromarray(255 - diff_np.astype(np.uint8))
            diff_segment = streamlit_util.audio_segment_from_spectrogram_image(
                image=diff_image,
                params=params,
                device=device,
            )

            audio_bytes = io.BytesIO()
            diff_segment.export(audio_bytes, format="wav")
            st.audio(audio_bytes)

    # Combine clips with a crossfade based on overlap
    crossfade_ms = int(overlap_duration_s * 1000)
    combined_segment = result_segments[0]
    for segment in result_segments[1:]:
        combined_segment = combined_segment.append(segment, crossfade=crossfade_ms)

    audio_bytes = io.BytesIO()
    combined_segment.export(audio_bytes, format="mp3")
    st.write(f"#### Final Audio ({combined_segment.duration_seconds}s)")
    st.audio(audio_bytes, format="audio/mp3")


@st.cache
def test(segment: pydub.AudioSegment, counter: int) -> int:
    st.write("#### Trimmed")
    st.write(segment.duration_seconds)
    return counter


if __name__ == "__main__":
    render_audio_to_audio()

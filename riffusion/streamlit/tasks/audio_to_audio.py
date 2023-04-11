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
from riffusion.streamlit.tasks.interpolation import get_prompt_inputs, run_interpolation
from riffusion.util import audio_util


def render() -> None:
    st.subheader("✨ Audio to Audio")
    st.write(
        """
    Modify existing audio from a text prompt or interpolate between two.
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

            If the Interpolation check box is enabled, supports entering two sets of prompt,
            seed, and denoising value and smoothly blends between them along the selected
            duration of the audio. This is a great way to create a transition.
            """
        )

    device = streamlit_util.select_device(st.sidebar)
    extension = streamlit_util.select_audio_extension(st.sidebar)
    checkpoint = streamlit_util.select_checkpoint(st.sidebar)

    use_20k = st.sidebar.checkbox("Use 20kHz", value=False)
    use_magic_mix = st.sidebar.checkbox("Use Magic Mix", False)

    with st.sidebar:
        num_inference_steps = T.cast(
            int,
            st.number_input(
                "Steps per sample", value=25, help="Number of denoising steps per model run"
            ),
        )

        guidance = st.number_input(
            "Guidance",
            value=7.0,
            help="How much the model listens to the text prompt",
        )

        scheduler = st.selectbox(
            "Scheduler",
            options=streamlit_util.SCHEDULER_OPTIONS,
            index=0,
            help="Which diffusion scheduler to use",
        )
        assert scheduler is not None

    audio_file = st.file_uploader(
        "Upload audio",
        type=streamlit_util.AUDIO_EXTENSIONS,
        label_visibility="collapsed",
    )

    if not audio_file:
        st.info("Upload audio to get started")
        return

    st.write("#### Original")
    st.audio(audio_file)

    segment = streamlit_util.load_audio_file(audio_file)

    # TODO(hayk): Fix
    if segment.frame_rate != 44100:
        st.warning("Audio must be 44100Hz. Converting")
        segment = segment.set_frame_rate(44100)
    st.write(f"Duration: {segment.duration_seconds:.2f}s, Sample Rate: {segment.frame_rate}Hz")

    clip_p = get_clip_params()
    start_time_s = clip_p["start_time_s"]
    clip_duration_s = clip_p["clip_duration_s"]
    overlap_duration_s = clip_p["overlap_duration_s"]

    duration_s = min(clip_p["duration_s"], segment.duration_seconds - start_time_s)
    increment_s = clip_duration_s - overlap_duration_s
    clip_start_times = start_time_s + np.arange(0, duration_s - clip_duration_s, increment_s)

    write_clip_details(
        clip_start_times=clip_start_times,
        clip_duration_s=clip_duration_s,
        overlap_duration_s=overlap_duration_s,
    )

    interpolate = st.checkbox(
        "Interpolate between two endpoints",
        value=False,
        help="Interpolate between two prompts, seeds, or denoising values along the"
        "duration of the segment",
    )

    counter = streamlit_util.StreamlitCounter()

    denoising_default = 0.55
    with st.form("audio to audio form"):
        if interpolate:
            left, right = st.columns(2)

            with left:
                st.write("##### Prompt A")
                prompt_input_a = PromptInput(
                    guidance=guidance,
                    **get_prompt_inputs(key="a", denoising_default=denoising_default),
                )

            with right:
                st.write("##### Prompt B")
                prompt_input_b = PromptInput(
                    guidance=guidance,
                    **get_prompt_inputs(key="b", denoising_default=denoising_default),
                )
        elif use_magic_mix:
            prompt = st.text_input("Prompt", key="prompt_a")

            row = st.columns(4)

            seed = T.cast(
                int,
                row[0].number_input(
                    "Seed",
                    value=42,
                    key="seed_a",
                ),
            )
            prompt_input_a = PromptInput(
                prompt=prompt,
                seed=seed,
                guidance=guidance,
            )
            magic_mix_kmin = row[1].number_input("Kmin", value=0.3)
            magic_mix_kmax = row[2].number_input("Kmax", value=0.5)
            magic_mix_mix_factor = row[3].number_input("Mix Factor", value=0.5)
        else:
            prompt_input_a = PromptInput(
                guidance=guidance,
                **get_prompt_inputs(
                    key="a",
                    include_negative_prompt=True,
                    cols=True,
                    denoising_default=denoising_default,
                ),
            )

        st.form_submit_button("Riff", type="primary", on_click=counter.increment)

    show_clip_details = st.sidebar.checkbox("Show Clip Details", True)
    show_difference = st.sidebar.checkbox("Show Difference", False)

    clip_segments = slice_audio_into_clips(
        segment=segment,
        clip_start_times=clip_start_times,
        clip_duration_s=clip_duration_s,
    )

    if not prompt_input_a.prompt:
        st.info("Enter a prompt")
        return

    if counter.value == 0:
        return

    st.write(f"## Counter: {counter.value}")

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

    if interpolate:
        # TODO(hayk): Make not linspace
        alphas = list(np.linspace(0, 1, len(clip_segments)))
        alphas_str = ", ".join([f"{alpha:.2f}" for alpha in alphas])
        st.write(f"**Alphas** : [{alphas_str}]")

    result_images: T.List[Image.Image] = []
    result_segments: T.List[pydub.AudioSegment] = []
    for i, clip_segment in enumerate(clip_segments):
        st.write(f"### Clip {i} at {clip_start_times[i]:.2f}s")

        audio_bytes = io.BytesIO()
        clip_segment.export(audio_bytes, format="wav")

        init_image = streamlit_util.spectrogram_image_from_audio(
            clip_segment,
            params=params,
            device=device,
        )

        # TODO(hayk): Roll this into spectrogram_image_from_audio?
        init_image_resized = scale_image_to_32_stride(init_image)

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

        if interpolate:
            assert use_magic_mix is False, "Cannot use magic mix and interpolate together"
            inputs = InferenceInput(
                alpha=float(alphas[i]),
                num_inference_steps=num_inference_steps,
                seed_image_id="og_beat",
                start=prompt_input_a,
                end=prompt_input_b,
            )

            image, audio_bytes = run_interpolation(
                inputs=inputs,
                init_image=init_image_resized,
                device=device,
                checkpoint=checkpoint,
            )
        elif use_magic_mix:
            assert not prompt_input_a.negative_prompt, "No negative prompt with magic mix"
            image = streamlit_util.run_img2img_magic_mix(
                prompt=prompt_input_a.prompt,
                init_image=init_image_resized,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance,
                seed=prompt_input_a.seed,
                kmin=magic_mix_kmin,
                kmax=magic_mix_kmax,
                mix_factor=magic_mix_mix_factor,
                device=device,
                scheduler=scheduler,
                checkpoint=checkpoint,
            )
        else:
            image = streamlit_util.run_img2img(
                prompt=prompt_input_a.prompt,
                init_image=init_image_resized,
                denoising_strength=prompt_input_a.denoising,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance,
                negative_prompt=prompt_input_a.negative_prompt,
                seed=prompt_input_a.seed,
                progress_callback=progress_callback,
                device=device,
                scheduler=scheduler,
                checkpoint=checkpoint,
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
            diff_segment.export(audio_bytes, format=extension)
            st.audio(audio_bytes)

    # Combine clips with a crossfade based on overlap
    combined_segment = audio_util.stitch_segments(result_segments, crossfade_s=overlap_duration_s)

    st.write(f"#### Final Audio ({combined_segment.duration_seconds}s)")

    input_name = Path(audio_file.name).stem
    output_name = f"{input_name}_{prompt_input_a.prompt.replace(' ', '_')}"
    streamlit_util.display_and_download_audio(combined_segment, output_name, extension=extension)


def get_clip_params(advanced: bool = False) -> T.Dict[str, T.Any]:
    """
    Render the parameters of slicing audio into clips.
    """
    p: T.Dict[str, T.Any] = {}

    cols = st.columns(4)

    p["start_time_s"] = cols[0].number_input(
        "Start Time [s]",
        min_value=0.0,
        value=0.0,
    )
    p["duration_s"] = cols[1].number_input(
        "Duration [s]",
        min_value=0.0,
        value=20.0,
    )

    if advanced:
        p["clip_duration_s"] = cols[2].number_input(
            "Clip Duration [s]",
            min_value=3.0,
            max_value=10.0,
            value=5.0,
        )
    else:
        p["clip_duration_s"] = 5.0

    if advanced:
        p["overlap_duration_s"] = cols[3].number_input(
            "Overlap Duration [s]",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
        )
    else:
        p["overlap_duration_s"] = 0.2

    return p


def write_clip_details(
    clip_start_times: np.ndarray, clip_duration_s: float, overlap_duration_s: float
):
    """
    Write details of the clips to be sliced from an audio segment.
    """
    clip_details_text = (
        f"Slicing {len(clip_start_times)} clips of duration {clip_duration_s}s "
        f"with overlap {overlap_duration_s}s"
    )

    with st.expander(clip_details_text):
        st.dataframe(
            {
                "Start Time [s]": clip_start_times,
                "End Time [s]": clip_start_times + clip_duration_s,
                "Duration [s]": clip_duration_s,
            }
        )


def slice_audio_into_clips(
    segment: pydub.AudioSegment, clip_start_times: T.Sequence[float], clip_duration_s: float
) -> T.List[pydub.AudioSegment]:
    """
    Slice an audio segment into a list of clips of a given duration at the given start times.
    """
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

    return clip_segments


def scale_image_to_32_stride(image: Image.Image) -> Image.Image:
    """
    Scale an image to a size that is a multiple of 32.
    """
    closest_width = int(np.ceil(image.width / 32) * 32)
    closest_height = int(np.ceil(image.height / 32) * 32)
    return image.resize((closest_width, closest_height), Image.BICUBIC)

import tempfile
import typing as T
from pathlib import Path

import numpy as np
import pydub
import streamlit as st

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


def render() -> None:
    st.subheader("📎 Sample Clips")
    st.write(
        """
    Export short clips from an audio file.
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            This tool simply allows uploading an audio file and randomly sampling short clips
            from it. It's useful for generating a large number of short clips from a single
            audio file. Outputs can be saved to a given directory with a given audio extension.
            """
        )

    audio_file = st.file_uploader(
        "Upload a file",
        type=streamlit_util.AUDIO_EXTENSIONS,
        label_visibility="collapsed",
    )
    if not audio_file:
        st.info("Upload an audio file to get started")
        return

    st.audio(audio_file)

    segment = pydub.AudioSegment.from_file(audio_file)
    st.write(
        "  \n".join(
            [
                f"**Duration**: {segment.duration_seconds:.3f} seconds",
                f"**Channels**: {segment.channels}",
                f"**Sample rate**: {segment.frame_rate} Hz",
                f"**Sample width**: {segment.sample_width} bytes",
            ]
        )
    )

    device = streamlit_util.select_device(st.sidebar)
    extension = streamlit_util.select_audio_extension(st.sidebar)
    save_to_disk = st.sidebar.checkbox("Save to Disk", False)
    export_as_mono = st.sidebar.checkbox("Export as Mono", False)
    compute_spectrograms = st.sidebar.checkbox("Compute Spectrograms", False)

    row = st.columns(4)
    num_clips = T.cast(int, row[0].number_input("Number of Clips", value=3))
    duration_ms = T.cast(int, row[1].number_input("Duration (ms)", value=5000))
    seed = T.cast(int, row[2].number_input("Seed", value=42))

    counter = streamlit_util.StreamlitCounter()
    st.button("Sample Clips", type="primary", on_click=counter.increment)
    if counter.value == 0:
        return

    # Optionally pick an output directory
    if save_to_disk:
        output_dir = tempfile.mkdtemp(prefix="sample_clips_")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        st.info(f"Output directory: `{output_dir}`")

        if compute_spectrograms:
            images_dir = output_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

    if seed >= 0:
        np.random.seed(seed)

    if export_as_mono and segment.channels > 1:
        segment = segment.set_channels(1)

    if save_to_disk:
        st.info(f"Writing {num_clips} clip(s) to `{str(output_path)}`")

    # TODO(hayk): Share code with riffusion.cli.sample_clips.
    segment_duration_ms = int(segment.duration_seconds * 1000)
    for i in range(num_clips):
        clip_start_ms = np.random.randint(0, segment_duration_ms - duration_ms)
        clip = segment[clip_start_ms : clip_start_ms + duration_ms]

        clip_name = f"clip_{i}_start_{clip_start_ms}_ms_duration_{duration_ms}_ms"

        st.write(f"#### Clip {i + 1} / {num_clips} -- `{clip_name}`")

        streamlit_util.display_and_download_audio(
            clip,
            name=clip_name,
            extension=extension,
        )

        if save_to_disk:
            clip_path = output_path / f"{clip_name}.{extension}"
            clip.export(clip_path, format=extension)

        if compute_spectrograms:
            params = SpectrogramParams()

            image = streamlit_util.spectrogram_image_from_audio(
                clip,
                params=params,
                device=device,
            )

            st.image(image)

            if save_to_disk:
                image_path = images_dir / f"{clip_name}.jpeg"
                image.save(image_path)

    if save_to_disk:
        st.info(f"Wrote {num_clips} clip(s) to `{str(output_path)}`")

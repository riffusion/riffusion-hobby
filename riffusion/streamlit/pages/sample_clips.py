import tempfile
import typing as T
from pathlib import Path

import numpy as np
import pydub
import streamlit as st


def render_sample_clips() -> None:
    st.set_page_config(layout="wide", page_icon="🎸")

    st.subheader(":scissors: Sample Clips")
    st.write(
        """
    Export short clips from an audio file.
    """
    )

    audio_file = st.file_uploader(
        "Upload a file",
        type=["wav", "mp3", "ogg"],
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

    seed = T.cast(int, st.sidebar.number_input("Seed", value=42))
    duration_ms = T.cast(int, st.sidebar.number_input("Duration (ms)", value=5000))
    export_as_mono = st.sidebar.checkbox("Export as Mono", False)
    num_clips = T.cast(int, st.sidebar.number_input("Number of Clips", value=3))
    extension = st.sidebar.selectbox("Extension", ["mp3", "wav", "ogg"])
    assert extension is not None

    # Optionally specify an output directory
    output_dir = st.text_input("Output Directory")
    if not output_dir:
        tmp_dir = tempfile.mkdtemp(prefix="sample_clips_")
        st.info(f"Specify an output directory. Suggested: `{tmp_dir}`")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if seed >= 0:
        np.random.seed(seed)

    if export_as_mono and segment.channels > 1:
        segment = segment.set_channels(1)

    # TODO(hayk): Share code with riffusion.cli.sample_clips.
    segment_duration_ms = int(segment.duration_seconds * 1000)
    for i in range(num_clips):
        clip_start_ms = np.random.randint(0, segment_duration_ms - duration_ms)
        clip = segment[clip_start_ms : clip_start_ms + duration_ms]

        clip_name = f"clip_{i}_start_{clip_start_ms}_ms_duration_{duration_ms}_ms.{extension}"

        st.write(f"#### Clip {i + 1} / {num_clips} -- `{clip_name}`")

        clip_path = output_path / clip_name
        clip.export(clip_path, format=extension)

        st.audio(str(clip_path))

    st.info(f"Wrote {num_clips} clip(s) to `{str(output_path)}`")


if __name__ == "__main__":
    render_sample_clips()

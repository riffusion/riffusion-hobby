import typing as T
from pathlib import Path

import pydub
import streamlit as st

from riffusion.audio_splitter import split_audio
from riffusion.streamlit import util as streamlit_util
from riffusion.util import audio_util


def render() -> None:
    st.subheader("✂️ Audio Splitter")
    st.write(
        """
    Split audio into individual instrument stems.
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            This tool allows uploading an audio file of arbitrary length and splits it into
            stems of vocals, drums, bass, and other. It does this using a deep network that
            sweeps over the audio in clips, extracts the stems, and then cross fades the clips
            back together to construct the full length stems. It's particularly useful in
            combination with audio_to_audio, for example to split and preserve vocals while
            modifying the rest of the track with a prompt. Or, to pull out drums to add later
            in a DAW.
            """
        )

    device = streamlit_util.select_device(st.sidebar)

    extension_options = ["mp3", "wav", "m4a", "ogg", "flac", "webm"]
    extension = st.sidebar.selectbox(
        "Output format",
        options=extension_options,
        index=extension_options.index("mp3"),
    )
    assert extension is not None

    audio_file = st.file_uploader(
        "Upload audio",
        type=extension_options,
        label_visibility="collapsed",
    )

    stem_options = ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"]
    recombine = st.sidebar.multiselect(
        "Recombine",
        options=stem_options,
        default=[],
        help="Recombine these stems at the end",
    )

    if not audio_file:
        st.info("Upload audio to get started")
        return

    st.write("#### Original")
    st.audio(audio_file)

    counter = streamlit_util.StreamlitCounter()
    st.button("Split", type="primary", on_click=counter.increment)
    if counter.value == 0:
        return

    segment = streamlit_util.load_audio_file(audio_file)

    # Split
    stems = split_audio_cached(segment, device=device)

    input_name = Path(audio_file.name).stem

    # Display each
    for name in stem_options:
        stem = stems[name.lower()]
        st.write(f"#### Stem: {name}")

        output_name = f"{input_name}_{name.lower()}"
        streamlit_util.display_and_download_audio(stem, output_name, extension=extension)

    if recombine:
        recombine_lower = [r.lower() for r in recombine]
        segments = [s for name, s in stems.items() if name in recombine_lower]
        recombined = audio_util.overlay_segments(segments)

        # Display
        st.write(f"#### Recombined: {', '.join(recombine)}")
        output_name = f"{input_name}_{'_'.join(recombine_lower)}"
        streamlit_util.display_and_download_audio(recombined, output_name, extension=extension)


@st.cache
def split_audio_cached(
    segment: pydub.AudioSegment, device: str = "cuda"
) -> T.Dict[str, pydub.AudioSegment]:
    return split_audio(segment, device=device)

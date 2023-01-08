import io

import pydub
import streamlit as st

from riffusion.audio_splitter import split_audio
from riffusion.streamlit import util as streamlit_util


def render_split_audio() -> None:
    st.set_page_config(layout="wide", page_icon="🎸")

    st.subheader(":scissors: Audio Splitter")
    st.write(
        """
    Split an audio into stems of {vocals, drums, bass, other}.
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

    audio_file = st.file_uploader(
        "Upload audio",
        type=["mp3", "m4a", "ogg", "wav", "flac", "webm"],
        label_visibility="collapsed",
    )

    stem_options = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    recombine = st.sidebar.multiselect(
        "Recombine",
        options=stem_options,
        default=[],
        help="Recombine these stems at the end",
    )

    if not audio_file:
        st.info("Upload audio to get started")
        return

    st.write("#### original")
    # TODO(hayk): This might be bogus, it can be other formats..
    st.audio(audio_file, format="audio/mp3")

    if not st.button("Split", type="primary"):
        return

    segment = streamlit_util.load_audio_file(audio_file)

    # Split
    stems = split_audio(segment, device=device)

    # Display each
    for name, stem in stems.items():
        st.write(f"#### {name}")
        audio_bytes = io.BytesIO()
        stem.export(audio_bytes, format="mp3")
        st.audio(audio_bytes, format="audio/mp3")

    if recombine:
        recombined: pydub.AudioSegment = None
        for name, stem in stems.items():
            if name in recombine:
                if recombined is None:
                    recombined = stem
                else:
                    recombined = recombined.overlay(stem)

        # Display
        st.write("#### recombined")
        audio_bytes = io.BytesIO()
        recombined.export(audio_bytes, format="mp3")
        st.audio(audio_bytes, format="audio/mp3")


if __name__ == "__main__":
    render_split_audio()

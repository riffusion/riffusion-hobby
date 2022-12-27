import pydub
import streamlit as st


def run():
    st.set_page_config(layout="wide", page_icon="🎸")

    audio_file = st.file_uploader("Upload a file", type=["wav", "mp3", "ogg"])
    if not audio_file:
        st.info("Upload an audio file to get started")
        return

    st.audio(audio_file)

    segment = pydub.AudioSegment.from_file(audio_file)
    st.write("  \n".join([
        f"**Duration**: {segment.duration_seconds:.3f} seconds",
        f"**Channels**: {segment.channels}",
        f"**Sample rate**: {segment.frame_rate} Hz",
        f"**Sample width**: {segment.sample_width} bytes",
    ]))


if __name__ == "__main__":
    run()

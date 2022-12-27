import dataclasses

import streamlit as st
from PIL import Image

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util
from riffusion.util.image_util import exif_from_image


def render_image_to_audio() -> None:
    st.set_page_config(layout="wide", page_icon="🎸")

    st.subheader(":musical_keyboard: Image to Audio")
    st.write(
        """
    Reconstruct audio from spectrogram images.
    """
    )

    device = streamlit_util.select_device(st.sidebar)

    image_file = st.file_uploader(
        "Upload a file",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if not image_file:
        st.info("Upload an image file to get started")
        return

    image = Image.open(image_file)
    st.image(image)

    with st.expander("Image metadata", expanded=False):
        exif = exif_from_image(image)
        st.json(exif)

    try:
        params = SpectrogramParams.from_exif(exif=image.getexif())
    except KeyError:
        st.info("Could not find spectrogram parameters in exif data. Using defaults.")
        params = SpectrogramParams()

    with st.expander("Spectrogram Parameters", expanded=False):
        st.json(dataclasses.asdict(params))

    audio_bytes = streamlit_util.audio_bytes_from_spectrogram_image(
        image=image.copy(),
        params=params,
        device=device,
        output_format="mp3",
    )
    st.audio(audio_bytes)


if __name__ == "__main__":
    render_image_to_audio()

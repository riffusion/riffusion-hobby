import io

import streamlit as st
from PIL import Image

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util
from riffusion.util.image_util import exif_from_image


def render_image_to_audio() -> None:
    image_file = st.sidebar.file_uploader(
        "Upload a file",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if not image_file:
        st.info("Upload an image file to get started")
        return

    image = Image.open(image_file)
    st.image(image)

    exif = exif_from_image(image)
    st.write("Exif data:")
    st.write(exif)

    device = "cuda"

    try:
        params = SpectrogramParams.from_exif(exif=image.getexif())
    except KeyError:
        st.warning("Could not find spectrogram parameters in exif data. Using defaults.")
        params = SpectrogramParams()

    # segment = streamlit_util.audio_from_spectrogram_image(
    #     image=image,
    #     params=params,
    #     device=device,
    # )

    # mp3_bytes = io.BytesIO()
    # segment.export(mp3_bytes, format="mp3")
    # mp3_bytes.seek(0)

    # st.audio(mp3_bytes)


if __name__ == "__main__":
    render_image_to_audio()

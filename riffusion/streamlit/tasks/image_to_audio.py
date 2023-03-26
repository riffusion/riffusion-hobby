import dataclasses
from pathlib import Path

import streamlit as st
from PIL import Image

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util
from riffusion.util.image_util import exif_from_image


def render() -> None:
    st.subheader("⏈ Image to Audio")
    st.write(
        """
    Reconstruct audio from spectrogram images.
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            This tool takes an existing spectrogram image and reconstructs it into an audio
            waveform. It also displays the EXIF metadata stored inside the image, which can
            contain the parameters used to create the spectrogram image. If no EXIF is contained,
            assumes default parameters.
            """
        )

    device = streamlit_util.select_device(st.sidebar)
    extension = streamlit_util.select_audio_extension(st.sidebar)

    use_20k = st.sidebar.checkbox("Use 20kHz", value=False)

    image_file = st.file_uploader(
        "Upload a file",
        type=streamlit_util.IMAGE_EXTENSIONS,
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
        if use_20k:
            params = SpectrogramParams(
                min_frequency=10,
                max_frequency=20000,
                stereo=True,
            )
        else:
            params = SpectrogramParams()

    with st.expander("Spectrogram Parameters", expanded=False):
        st.json(dataclasses.asdict(params))

    segment = streamlit_util.audio_segment_from_spectrogram_image(
        image=image.copy(),
        params=params,
        device=device,
    )

    streamlit_util.display_and_download_audio(
        segment,
        name=Path(image_file.name).stem,
        extension=extension,
    )

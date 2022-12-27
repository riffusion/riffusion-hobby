"""
Streamlit utilities (mostly cached wrappers around riffusion code).
"""

import pydub
import streamlit as st
from PIL import Image

from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams


@st.experimental_singleton
def load_riffusion_checkpoint(
    checkpoint: str = "riffusion/riffusion-model-v1",
    no_traced_unet: bool = False,
    device: str = "cuda",
) -> RiffusionPipeline:
    """
    Load the riffusion pipeline.
    """
    return RiffusionPipeline.load_checkpoint(
        checkpoint=checkpoint,
        use_traced_unet=not no_traced_unet,
        device=device,
    )

# class CachedSpectrogramImageConverter:

#     def __init__(self, params: SpectrogramParams, device: str = "cuda"):
#         self.p = params
#         self.device = device
#         self.converter = self._converter(params, device)

#     @staticmethod
#     @st.experimental_singleton
#     def _converter(params: SpectrogramParams, device: str) -> SpectrogramImageConverter:
#          return SpectrogramImageConverter(params=params, device=device)

#     def audio_from_spectrogram_image(
#         self,
#         image: Image.Image
#     ) -> pydub.AudioSegment:
#         return self._converter.audio_from_spectrogram_image(image)


@st.experimental_singleton
def spectrogram_image_converter(
    params: SpectrogramParams,
    device: str = "cuda",
) -> SpectrogramImageConverter:
    return SpectrogramImageConverter(params=params, device=device)


@st.experimental_memo
def audio_from_spectrogram_image(
    image: Image.Image,
    params: SpectrogramParams,
    device: str = "cuda",
) -> pydub.AudioSegment:
    converter = spectrogram_image_converter(params=params, device=device)
    return converter.audio_from_spectrogram_image(image)


# @st.experimental_memo
# def spectrogram_image_from_audio(
#     segment: pydub.AudioSegment,
#     params: SpectrogramParams,
#     device: str = "cuda",
# ) -> pydub.AudioSegment:
#     converter = spectrogram_image_converter(params=params, device=device)
#     return converter.spectrogram_image_from_audio(segment)

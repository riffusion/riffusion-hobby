import numpy as np
import pydub
from PIL import Image

from riffusion.spectrogram_converter import SpectrogramConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import image_util


class SpectrogramImageConverter:
    """
    Convert between spectrogram images and audio segments.

    This is a wrapper around SpectrogramConverter that additionally converts from spectrograms
    to images and back. The real audio processing lives in SpectrogramConverter.
    """

    def __init__(self, params: SpectrogramParams, device: str = "cuda"):
        self.p = params
        self.device = device
        self.converter = SpectrogramConverter(params=params, device=device)

    def spectrogram_image_from_audio(
        self,
        segment: pydub.AudioSegment,
    ) -> Image.Image:
        """
        Compute a spectrogram image from an audio segment.

        Args:
            segment: Audio segment to convert

        Returns:
            Spectrogram image (in pillow format)
        """
        assert int(segment.frame_rate) == self.p.sample_rate, "Sample rate mismatch"

        if self.p.stereo:
            if segment.channels == 1:
                print("WARNING: Mono audio but stereo=True, cloning channel")
                segment = segment.set_channels(2)
            elif segment.channels > 2:
                print("WARNING: Multi channel audio, reducing to stereo")
                segment = segment.set_channels(2)
        else:
            if segment.channels > 1:
                print("WARNING: Stereo audio but stereo=False, setting to mono")
                segment = segment.set_channels(1)

        spectrogram = self.converter.spectrogram_from_audio(segment)

        image = image_util.image_from_spectrogram(
            spectrogram,
            power=self.p.power_for_image,
        )

        # Store conversion params in exif metadata of the image
        exif_data = self.p.to_exif()
        exif_data[SpectrogramParams.ExifTags.MAX_VALUE.value] = float(np.max(spectrogram))
        exif = image.getexif()
        exif.update(exif_data.items())

        return image

    def audio_from_spectrogram_image(
        self,
        image: Image.Image,
        apply_filters: bool = True,
        max_value: float = 30e6,
    ) -> pydub.AudioSegment:
        """
        Reconstruct an audio segment from a spectrogram image.

        Args:
            image: Spectrogram image (in pillow format)
            apply_filters: Apply post-processing to improve the reconstructed audio
            max_value: Scaled max amplitude of the spectrogram. Shouldn't matter.
        """
        spectrogram = image_util.spectrogram_from_image(
            image,
            max_value=max_value,
            power=self.p.power_for_image,
            stereo=self.p.stereo,
        )

        segment = self.converter.audio_from_spectrogram(
            spectrogram,
            apply_filters=apply_filters,
        )

        return segment

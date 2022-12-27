import numpy as np
import pydub

from riffusion.spectrogram_converter import SpectrogramConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import image_util

from .test_case import TestCase


class ImageUtilTest(TestCase):
    """
    Test riffusion.util.image_util
    """

    def test_spectrogram_to_image_round_trip(self) -> None:
        audio_path = (
            self.TEST_DATA_PATH
            / "tired_traveler"
            / "clips"
            / "clip_2_start_103694_ms_duration_5678_ms.wav"
        )

        # Load up the audio file
        segment = pydub.AudioSegment.from_file(audio_path)

        # Convert to mono
        segment = segment.set_channels(1)

        # Compute a spectrogram with default params
        params = SpectrogramParams(sample_rate=segment.frame_rate)
        converter = SpectrogramConverter(params=params, device=self.DEVICE)
        spectrogram = converter.spectrogram_from_audio(segment)

        # Compute the image from the spectrogram
        image = image_util.image_from_spectrogram(
            spectrogram=spectrogram,
            power=params.power_for_image,
        )

        # Save the max value
        max_value = np.max(spectrogram)

        # Compute the spectrogram from the image
        spectrogram_reversed = image_util.spectrogram_from_image(
            image=image,
            max_value=max_value,
            power=params.power_for_image,
            stereo=params.stereo,
        )

        # Check the shapes
        self.assertEqual(spectrogram.shape, spectrogram_reversed.shape)

        # Check the max values
        self.assertEqual(np.max(spectrogram), np.max(spectrogram_reversed))

        # Check the median values
        self.assertTrue(
            np.allclose(np.median(spectrogram), np.median(spectrogram_reversed), rtol=0.05)
        )

        # Make sure all values are somewhat similar, but allow for discretization error
        # TODO(hayk): Investigate error more closely
        self.assertTrue(np.allclose(spectrogram, spectrogram_reversed, rtol=0.15))

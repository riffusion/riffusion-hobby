import typing as T

import numpy as np
from PIL import Image

from riffusion.cli import audio_to_image
from riffusion.spectrogram_params import SpectrogramParams

from .test_case import TestCase


class AudioToImageTest(TestCase):
    """
    Test riffusion.cli audio-to-image
    """

    @classmethod
    def default_params(cls) -> T.Dict:
        return dict(
            step_size_ms=10,
            num_frequencies=512,
            # TODO(hayk): Change these to [20, 20000] once a model is updated
            min_frequency=0,
            max_frequency=10000,
            stereo=False,
            device=cls.DEVICE,
        )

    def test_audio_to_image(self) -> None:
        """
        Test audio-to-image with default params.
        """
        params = self.default_params()
        self.helper_test_with_params(params)

    def test_stereo(self) -> None:
        """
        Test audio-to-image with stereo=True.
        """
        params = self.default_params()
        params["stereo"] = True
        self.helper_test_with_params(params)

    def helper_test_with_params(self, params: T.Dict) -> None:
        audio_path = (
            self.TEST_DATA_PATH
            / "tired_traveler"
            / "clips"
            / "clip_2_start_103694_ms_duration_5678_ms.wav"
        )
        output_dir = self.get_tmp_dir("audio_to_image_")

        if params["stereo"]:
            stem = f"{audio_path.stem}_stereo"
        else:
            stem = audio_path.stem

        image_path = output_dir / f"{stem}.png"

        audio_to_image(audio=str(audio_path), image=str(image_path), **params)

        # Check that the image exists
        self.assertTrue(image_path.exists())

        pil_image = Image.open(image_path)

        # Check the image mode
        self.assertEqual(pil_image.mode, "RGB")

        # Check the image dimensions
        duration_ms = 5678
        self.assertTrue(str(duration_ms) in audio_path.name)
        expected_image_width = round(duration_ms / params["step_size_ms"])
        self.assertEqual(pil_image.width, expected_image_width)
        self.assertEqual(pil_image.height, params["num_frequencies"])

        # Get channels as numpy arrays
        channels = [np.array(pil_image.getchannel(i)) for i in range(len(pil_image.getbands()))]
        self.assertEqual(len(channels), 3)

        if params["stereo"]:
            # Check that the first channel is zero
            self.assertTrue(np.all(channels[0] == 0))
        else:
            # Check that all channels are the same
            self.assertTrue(np.all(channels[0] == channels[1]))
            self.assertTrue(np.all(channels[0] == channels[2]))

        # Check that the image has exif data
        exif = pil_image.getexif()
        self.assertIsNotNone(exif)
        params_from_exif = SpectrogramParams.from_exif(exif)
        expected_params = SpectrogramParams(
            stereo=params["stereo"],
            step_size_ms=params["step_size_ms"],
            num_frequencies=params["num_frequencies"],
            max_frequency=params["max_frequency"],
        )
        self.assertTrue(params_from_exif == expected_params)

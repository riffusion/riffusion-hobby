import dataclasses
import typing as T

import pydub
from PIL import Image

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import fft_util

from .test_case import TestCase


class SpectrogramImageConverterTest(TestCase):
    """
    Test going from audio to spectrogram images to audio, testing the quality loss of the
    end-to-end pipeline.

    This test allows comparing multiple sets of spectrogram params by listening to output audio
    and by plotting their FFTs.

    See spectrogram_converter_test.py for a similar test that does not convert to images.
    """

    def test_round_trip(self) -> None:
        audio_path = (
            self.TEST_DATA_PATH
            / "tired_traveler"
            / "clips"
            / "clip_2_start_103694_ms_duration_5678_ms.wav"
        )
        output_dir = self.get_tmp_dir(prefix="spectrogram_image_round_trip_test_")

        # Load up the audio file
        segment = pydub.AudioSegment.from_file(audio_path)

        # Convert to mono if desired
        use_stereo = False
        if use_stereo:
            assert segment.channels == 2
        else:
            segment = segment.set_channels(1)

        # Define named sets of parameters
        param_sets: T.Dict[str, SpectrogramParams] = {}

        param_sets["default"] = SpectrogramParams(
            sample_rate=segment.frame_rate,
            stereo=use_stereo,
            step_size_ms=10,
            min_frequency=20,
            max_frequency=20000,
            num_frequencies=512,
        )

        if self.DEBUG:
            param_sets["freq_0_to_10k"] = dataclasses.replace(
                param_sets["default"],
                min_frequency=0,
                max_frequency=10000,
            )

        segments: T.Dict[str, pydub.AudioSegment] = {
            "original": segment,
        }
        images: T.Dict[str, Image.Image] = {}
        for name, params in param_sets.items():
            converter = SpectrogramImageConverter(params=params, device=self.DEVICE)
            images[name] = converter.spectrogram_image_from_audio(segment)
            segments[name] = converter.audio_from_spectrogram_image(
                image=images[name],
                apply_filters=True,
            )

        # Save images to disk
        for name, image in images.items():
            image_out = output_dir / f"{name}.png"
            image.save(image_out, exif=image.getexif(), format="PNG")
            print(f"Saved {image_out}")

        # Save segments to disk
        for name, segment in segments.items():
            audio_out = output_dir / f"{name}.wav"
            segment.export(audio_out, format="wav")
            print(f"Saved {audio_out}")

        # Check params
        self.assertEqual(segments["default"].channels, 2 if use_stereo else 1)
        self.assertEqual(segments["original"].channels, segments["default"].channels)
        self.assertEqual(segments["original"].frame_rate, segments["default"].frame_rate)
        self.assertEqual(segments["original"].sample_width, segments["default"].sample_width)

        # TODO(hayk): Test something more rigorous about the quality of the reconstruction.

        # If debugging, load up a browser tab plotting the FFTs
        if self.DEBUG:
            fft_util.plot_ffts(segments)

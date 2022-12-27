import typing as T

import pydub

from riffusion.cli import sample_clips

from .test_case import TestCase


class SampleClipsTest(TestCase):
    """
    Test riffusion.cli sample-clips
    """

    @staticmethod
    def default_params() -> T.Dict:
        return dict(
            num_clips=3,
            duration_ms=5678,
            mono=False,
            extension="wav",
            seed=42,
        )

    def test_sample_clips(self) -> None:
        """
        Test sample-clips with default params.
        """
        params = self.default_params()
        self.helper_test_with_params(params)

    def test_mono(self) -> None:
        """
        Test sample-clips with mono=True.
        """
        params = self.default_params()
        params["mono"] = True
        params["num_clips"] = 1
        self.helper_test_with_params(params)

    def test_mp3(self) -> None:
        """
        Test sample-clips with extension=mp3.
        """
        if pydub.AudioSegment.converter is None:
            self.skipTest("skipping, ffmpeg not found")

        params = self.default_params()
        params["extension"] = "mp3"
        params["num_clips"] = 1
        self.helper_test_with_params(params)

    def helper_test_with_params(self, params: T.Dict) -> None:
        """
        Test sample-clips with the given params.
        """
        audio_path = self.TEST_DATA_PATH / "tired_traveler" / "tired_traveler.mp3"
        output_dir = self.get_tmp_dir("sample_clips_")

        sample_clips(
            audio=str(audio_path),
            output_dir=str(output_dir),
            **params,
        )

        # For each file in output dir
        counter = 0
        for clip_path in output_dir.iterdir():
            # Check that it has the right extension
            self.assertEqual(clip_path.suffix, f".{params['extension']}")

            # Check that it has the right duration
            segment = pydub.AudioSegment.from_file(clip_path)
            self.assertEqual(round(segment.duration_seconds * 1000), params["duration_ms"])

            # Check that it has the right number of channels
            if params["mono"]:
                self.assertEqual(segment.channels, 1)
            else:
                self.assertEqual(segment.channels, 2)

            counter += 1

        self.assertEqual(counter, params["num_clips"])


if __name__ == "__main__":
    TestCase.main()

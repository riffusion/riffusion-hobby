from pathlib import Path

import pydub

from riffusion.cli import image_to_audio

from .test_case import TestCase


class ImageToAudioTest(TestCase):
    """
    Test riffusion.cli image-to-audio
    """

    def test_image_to_audio_mono(self) -> None:
        self.helper_image_to_audio(
            song_dir=self.TEST_DATA_PATH / "tired_traveler",
            clip_name="clip_2_start_103694_ms_duration_5678_ms",
            stereo=False,
        )

    def test_image_to_audio_stereo(self) -> None:
        self.helper_image_to_audio(
            song_dir=self.TEST_DATA_PATH / "tired_traveler",
            clip_name="clip_2_start_103694_ms_duration_5678_ms",
            stereo=True,
        )

    def helper_image_to_audio(self, song_dir: Path, clip_name: str, stereo: bool) -> None:
        if stereo:
            image_stem = clip_name + "_stereo"
        else:
            image_stem = clip_name

        image_path = song_dir / "images" / f"{image_stem}.png"
        output_dir = self.get_tmp_dir("image_to_audio_")
        audio_path = output_dir / f"{image_path.stem}.wav"

        image_to_audio(
            image=str(image_path),
            audio=str(audio_path),
            device=self.DEVICE,
        )

        # Check that the audio exists
        self.assertTrue(audio_path.exists())

        # Load the reconstructed audio and the original clip
        segment = pydub.AudioSegment.from_file(str(audio_path))
        expected_segment = pydub.AudioSegment.from_file(
            str(song_dir / "clips" / f"{clip_name}.wav")
        )

        # Check sample rate
        self.assertEqual(segment.frame_rate, expected_segment.frame_rate)

        # Check duration
        actual_duration_ms = round(segment.duration_seconds * 1000)
        expected_duration_ms = round(expected_segment.duration_seconds * 1000)
        self.assertTrue(abs(actual_duration_ms - expected_duration_ms) < 10)

        # Check the number of channels
        self.assertEqual(expected_segment.channels, 2)
        if stereo:
            self.assertEqual(segment.channels, 2)
        else:
            self.assertEqual(segment.channels, 1)


if __name__ == "__main__":
    TestCase.main()

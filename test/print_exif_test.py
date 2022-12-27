import contextlib
import io

from riffusion.cli import print_exif

from .test_case import TestCase


class PrintExifTest(TestCase):
    """
    Test riffusion.cli print-exif
    """

    def test_print_exif(self) -> None:
        """
        Test print-exif.
        """
        image_path = (
            self.TEST_DATA_PATH
            / "tired_traveler"
            / "images"
            / "clip_2_start_103694_ms_duration_5678_ms.png"
        )

        # Redirect stdout
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            print_exif(image=str(image_path))

        # Check that a couple of values are printed
        self.assertTrue("NUM_FREQUENCIES      =             512" in stdout.getvalue())
        self.assertTrue("SAMPLE_RATE          =           44100" in stdout.getvalue())

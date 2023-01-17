import os
import shutil
import tempfile
import typing as T
import unittest
import warnings
from pathlib import Path


class TestCase(unittest.TestCase):
    """
    Base class for tests.
    """

    # Where checked-in test data is stored
    TEST_DATA_PATH = Path(__file__).resolve().parent / "test_data"

    # Whether to run tests in debug mode (e.g. don't clean up temporary directories, show plots)
    DEBUG = bool(os.environ.get("RIFFUSION_TEST_DEBUG"))

    # Which torch device to use for tests
    DEVICE = os.environ.get("RIFFUSION_TEST_DEVICE", "cuda")

    @staticmethod
    def main(*args: T.Any, **kwargs: T.Any) -> None:
        """
        Run the tests.
        """
        unittest.main(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=ResourceWarning)

    def get_tmp_dir(self, prefix: str) -> Path:
        """
        Create a temporary directory.
        """
        tmp_dir = tempfile.mkdtemp(prefix=prefix)

        # Clean up the temporary directory if not debugging
        if not self.DEBUG:
            self.addCleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))

        dir_path = Path(tmp_dir)
        assert dir_path.is_dir()

        return dir_path

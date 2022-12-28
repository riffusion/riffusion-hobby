import subprocess
from pathlib import Path

from .test_case import TestCase


class LinterTest(TestCase):
    """
    Test that ruff, black, and mypy run cleanly.
    """

    HOME = Path(__file__).parent.parent

    def test_ruff(self) -> None:
        code = subprocess.check_call(["ruff", str(self.HOME)])
        self.assertEqual(code, 0)

    def test_black(self) -> None:
        code = subprocess.check_call(["black", "--check", str(self.HOME)])
        self.assertEqual(code, 0)

    def test_mypy(self) -> None:
        code = subprocess.check_call(["mypy", str(self.HOME)])
        self.assertEqual(code, 0)

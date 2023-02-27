"""Entrypoint script for Riffusion playground streamlit app."""
import os
import sys

from streamlit.web import cli as streamlit_cli


def main():
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(os.path.dirname(__file__), "streamlit", "playground.py"),
    ]
    sys.exit(streamlit_cli.main())


if __name__ == "__main__":
    main()

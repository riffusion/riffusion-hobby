"""
Command line tools for riffusion.
"""

from pathlib import Path

import argh
import numpy as np
import pydub
from PIL import Image

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import image_util


@argh.arg("--step-size-ms", help="Duration of one pixel in the X axis of the spectrogram image")
@argh.arg("--num-frequencies", help="Number of Y axes in the spectrogram image")
def audio_to_image(
    *,
    audio: str,
    image: str,
    step_size_ms: int = 10,
    num_frequencies: int = 512,
    min_frequency: int = 0,
    max_frequency: int = 10000,
    window_duration_ms: int = 100,
    padded_duration_ms: int = 400,
    power_for_image: float = 0.25,
    stereo: bool = False,
    device: str = "cuda",
):
    """
    Compute a spectrogram image from a waveform.
    """
    segment = pydub.AudioSegment.from_file(audio)

    params = SpectrogramParams(
        sample_rate=segment.frame_rate,
        stereo=stereo,
        window_duration_ms=window_duration_ms,
        padded_duration_ms=padded_duration_ms,
        step_size_ms=step_size_ms,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        num_frequencies=num_frequencies,
        power_for_image=power_for_image,
    )

    converter = SpectrogramImageConverter(params=params, device=device)

    pil_image = converter.spectrogram_image_from_audio(segment)

    pil_image.save(image, exif=pil_image.getexif(), format="PNG")
    print(f"Wrote {image}")


def print_exif(*, image: str) -> None:
    """
    Print the params of a spectrogram image as saved in the exif data.
    """
    pil_image = Image.open(image)
    exif_data = image_util.exif_from_image(pil_image)

    for name, value in exif_data.items():
        print(f"{name:<20} = {value:>15}")


def image_to_audio(*, image: str, audio: str, device: str = "cuda"):
    """
    Reconstruct an audio clip from a spectrogram image.
    """
    pil_image = Image.open(image)

    # Get parameters from image exif
    img_exif = pil_image.getexif()
    assert img_exif is not None

    try:
        params = SpectrogramParams.from_exif(exif=img_exif)
    except (KeyError, AttributeError):
        print("WARNING: Could not find spectrogram parameters in exif data. Using defaults.")
        params = SpectrogramParams()

    converter = SpectrogramImageConverter(params=params, device=device)
    segment = converter.audio_from_spectrogram_image(pil_image)

    extension = Path(audio).suffix[1:]
    segment.export(audio, format=extension)

    print(f"Wrote {audio} ({segment.duration_seconds:.2f} seconds)")


def sample_clips(
    *,
    audio: str,
    output_dir: str,
    num_clips: int = 1,
    duration_ms: int = 5000,
    mono: bool = False,
    extension: str = "wav",
    seed: int = -1,
):
    """
    Slice an audio file into clips of the given duration.
    """
    if seed >= 0:
        np.random.seed(seed)

    segment = pydub.AudioSegment.from_file(audio)

    if mono:
        segment = segment.set_channels(1)

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    segment_duration_ms = int(segment.duration_seconds * 1000)
    for i in range(num_clips):
        clip_start_ms = np.random.randint(0, segment_duration_ms - duration_ms)
        clip = segment[clip_start_ms : clip_start_ms + duration_ms]

        clip_name = f"clip_{i}_start_{clip_start_ms}_ms_duration_{duration_ms}_ms.{extension}"
        clip_path = output_dir_path / clip_name
        clip.export(clip_path, format=extension)
        print(f"Wrote {clip_path}")


def main():
    """
    Main entry point for the command line interface.
    """
    argh.dispatch_commands(
        [
            audio_to_image,
            image_to_audio,
            sample_clips,
            print_exif,
        ]
    )


if __name__ == "__main__":
    main()

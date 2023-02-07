"""
Command line tools for riffusion.
"""

import random
import typing as T
from multiprocessing.pool import ThreadPool
from pathlib import Path

import argh
import numpy as np
import pydub
import tqdm
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
    duration_ms: int = 5120,
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


def audio_to_images_batch(
    *,
    audio_dir: str,
    output_dir: str,
    image_extension: str = "jpg",
    step_size_ms: int = 10,
    num_frequencies: int = 512,
    min_frequency: int = 0,
    max_frequency: int = 10000,
    power_for_image: float = 0.25,
    mono: bool = False,
    sample_rate: int = 44100,
    device: str = "cuda",
    num_threads: T.Optional[int] = None,
    limit: int = -1,
):
    """
    Process audio clips into spectrograms in batch, multi-threaded.
    """
    audio_paths = list(Path(audio_dir).glob("*"))
    audio_paths.sort()

    if limit > 0:
        audio_paths = audio_paths[:limit]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    params = SpectrogramParams(
        step_size_ms=step_size_ms,
        num_frequencies=num_frequencies,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        power_for_image=power_for_image,
        stereo=not mono,
        sample_rate=sample_rate,
    )

    converter = SpectrogramImageConverter(params=params, device=device)

    def process_one(audio_path: Path) -> None:
        # Load
        try:
            segment = pydub.AudioSegment.from_file(str(audio_path))
        except Exception:
            return

        # TODO(hayk): Sanity checks on clip

        if mono and segment.channels != 1:
            segment = segment.set_channels(1)
        elif not mono and segment.channels != 2:
            segment = segment.set_channels(2)

        # Frame rate
        if segment.frame_rate != params.sample_rate:
            segment = segment.set_frame_rate(params.sample_rate)

        # Convert
        image = converter.spectrogram_image_from_audio(segment)

        # Save
        image_path = output_path / f"{audio_path.stem}.{image_extension}"
        image_format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}[image_extension]
        image.save(image_path, exif=image.getexif(), format=image_format)

    # Create thread pool
    pool = ThreadPool(processes=num_threads)
    with tqdm.tqdm(total=len(audio_paths)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(process_one, audio_paths)):
            pbar.update()


def sample_clips_batch(
    *,
    audio_dir: str,
    output_dir: str,
    num_clips_per_file: int = 1,
    duration_ms: int = 5120,
    mono: bool = False,
    extension: str = "mp3",
    num_threads: T.Optional[int] = None,
    glob: str = "*",
    limit: int = -1,
    seed: int = -1,
):
    """
    Sample short clips from a directory of audio files, multi-threaded.
    """
    audio_paths = list(Path(audio_dir).glob(glob))
    audio_paths.sort()

    # Exclude json
    audio_paths = [p for p in audio_paths if p.suffix != ".json"]

    if limit > 0:
        audio_paths = audio_paths[:limit]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if seed >= 0:
        random.seed(seed)

    def process_one(audio_path: Path) -> None:
        try:
            segment = pydub.AudioSegment.from_file(str(audio_path))
        except Exception:
            return

        if mono:
            segment = segment.set_channels(1)

        segment_duration_ms = int(segment.duration_seconds * 1000)
        for i in range(num_clips_per_file):
            try:
                clip_start_ms = np.random.randint(0, segment_duration_ms - duration_ms)
            except ValueError:
                continue

            clip = segment[clip_start_ms : clip_start_ms + duration_ms]

            clip_name = (
                f"{audio_path.stem}_{i}_"
                f"start_{clip_start_ms}_ms_dur_{duration_ms}_ms.{extension}"
            )
            clip.export(output_path / clip_name, format=extension)

    pool = ThreadPool(processes=num_threads)
    with tqdm.tqdm(total=len(audio_paths)) as pbar:
        for result in pool.imap_unordered(process_one, audio_paths):
            pbar.update()


if __name__ == "__main__":
    argh.dispatch_commands(
        [
            audio_to_image,
            image_to_audio,
            sample_clips,
            print_exif,
            audio_to_images_batch,
            sample_clips_batch,
        ]
    )

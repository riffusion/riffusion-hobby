"""
FFT tools to analyze frequency content of audio segments. This is not code for
dealing with spectrogram images, but for analysis of waveforms.
"""
import struct
import typing as T

import numpy as np
import plotly.graph_objects as go
import pydub
from scipy.fft import rfft, rfftfreq


def plot_ffts(
    segments: T.Dict[str, pydub.AudioSegment],
    title: str = "FFT",
    min_frequency: float = 20,
    max_frequency: float = 20000,
) -> None:
    """
    Plot an FFT analysis of the given audio segments.
    """
    ffts = {name: compute_fft(seg) for name, seg in segments.items()}

    fig = go.Figure(
        data=[go.Scatter(x=data[0], y=data[1], name=name) for name, data in ffts.items()],
        layout={"title": title},
    )
    fig.update_xaxes(
        range=[np.log(min_frequency) / np.log(10), np.log(max_frequency) / np.log(10)],
        type="log",
        title="Frequency",
    )
    fig.update_yaxes(title="Value")
    fig.show()


def compute_fft(sound: pydub.AudioSegment) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT of the given audio segment as a mono signal.

    Returns:
        frequencies: FFT computed frequencies
        amplitudes: Amplitudes of each frequency
    """
    # Convert to mono if needed.
    if sound.channels > 1:
        sound = sound.set_channels(1)

    sample_rate = sound.frame_rate

    num_samples = int(sound.frame_count())
    samples = struct.unpack(f"{num_samples * sound.channels}h", sound.raw_data)

    fft_values = rfft(samples)
    amplitudes = np.abs(fft_values)

    frequencies = rfftfreq(n=num_samples, d=1 / sample_rate)

    return frequencies, amplitudes

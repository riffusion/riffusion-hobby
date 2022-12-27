"""
Audio utility functions.
"""

import io

import numpy as np
import pydub
from scipy.io import wavfile


def audio_from_waveform(
    samples: np.ndarray, sample_rate: int, normalize: bool = False
) -> pydub.AudioSegment:
    """
    Convert a numpy array of samples of a waveform to an audio segment.
    """
    # Normalize volume to fit in int16
    if normalize:
        samples *= np.iinfo(np.int16).max / np.max(np.abs(samples))

    # Transpose and convert to int16
    samples = samples.transpose(1, 0)
    samples = samples.astype(np.int16)

    # Write to the bytes of a WAV file
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples)
    wav_bytes.seek(0)

    # Read into pydub
    return pydub.AudioSegment.from_wav(wav_bytes)


def apply_filters(segment: pydub.AudioSegment, compression: bool = False) -> pydub.AudioSegment:
    """
    Apply post-processing filters to the audio segment to compress it and
    keep at a -10 dBFS level.
    """
    # TODO(hayk): Come up with a principled strategy for these filters and experiment end-to-end.
    # TODO(hayk): Is this going to make audio unbalanced between sequential clips?

    if compression:
        segment = pydub.effects.normalize(
            segment,
            headroom=0.1,
        )

        segment = segment.apply_gain(-10 - segment.dBFS)

        # TODO(hayk): This is quite slow, ~1.7 seconds on a beefy CPU
        segment = pydub.effects.compress_dynamic_range(
            segment,
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
        )

    desired_db = -12
    segment = segment.apply_gain(desired_db - segment.dBFS)

    segment = pydub.effects.normalize(
        segment,
        headroom=0.1,
    )

    return segment

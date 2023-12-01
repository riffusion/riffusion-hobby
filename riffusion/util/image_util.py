"""
Module for converting between spectrograms tensors and spectrogram images, as well as
general helpers for operating on pillow images.
"""
import typing as T

import numpy as np
from PIL import Image

from riffusion.spectrogram_params import SpectrogramParams


def image_from_spectrogram(
    spectrogram: np.ndarray, power: float = 0.25, triple_res_mono=False
) -> Image.Image:
    """
    Compute a spectrogram image from a spectrogram magnitude array.

    This is the inverse of spectrogram_from_image, except for discretization error from
    quantizing to uint8.

    Args:
        spectrogram: (channels, frequency, time)
        power: A power curve to apply to the spectrogram to preserve contrast

    Returns:
        image: (frequency, time, channels)
    """
    # Rescale to 0-1
    max_value = np.max(spectrogram)
    data = spectrogram / max_value

    # Apply the power curve
    data = np.power(data, power)

    # Rescale to 0-255
    data = data * 255

    # Invert
    data = 255 - data

    # Convert to uint8
    data = data.astype(np.uint8)

    # Munge channels into a PIL image
    if data.shape[0] == 1:
        if triple_res_mono:
            # Temporarily transpose so that reshaping will order data such that
            # each RGB pixel will represent 3 consecutive frequency bins along frequency axis
            data = data.transpose(1, 0, 2)
            data = data.reshape(data.shape[0] // 3, 3, data.shape[2])
            data = data.transpose(0, 2, 1)
            image = Image.fromarray(data, mode="RGB")
        else:
            # TODO(hayk): Do we want to write single channel to disk instead?
            image = Image.fromarray(data[0], mode="L").convert("RGB")
    elif data.shape[0] == 2:
        data = np.array([np.zeros_like(data[0]), data[0], data[1]]).transpose(1, 2, 0)
        image = Image.fromarray(data, mode="RGB")
    else:
        raise NotImplementedError(f"Unsupported number of channels: {data.shape[0]}")

    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    return image


def spectrogram_from_image(
    image: Image.Image,
    power: float = 0.25,
    stereo: bool = False,
    triple_res_mono: bool = False,
    max_value: float = 30e6,
) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    This is the inverse of image_from_spectrogram, except for discretization error from
    quantizing to uint8.

    Args:
        image: (frequency, time, channels)
        power: The power curve applied to the spectrogram
        stereo: Whether the spectrogram encodes stereo data
        triple_res_mono: Whether the spectrogram uses R,G,B channels
            to encode triple resolution of frequency data in mono
        max_value: The max value of the original spectrogram. In practice doesn't matter.

    Returns:
        spectrogram: (channels, frequency, time)
    """
    # Convert to RGB if single channel
    if image.mode in ("P", "L"):
        image = image.convert("RGB")

    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    data = np.array(image)

    if triple_res_mono:
        data = data.transpose(0, 2, 1)
        data = data.reshape(1, data.shape[0] * data.shape[1], data.shape[2])
    else:
        # Munge channels into a numpy array of (channels, frequency, time)
        data = data.transpose(2, 0, 1)
        if stereo:
            # Take the G and B channels as done in image_from_spectrogram
            data = data[[1, 2], :, :]
        else:
            data = data[0:1, :, :]

    # Convert to floats
    data = data.astype(np.float32)

    # Invert
    data = 255 - data

    # Rescale to 0-1
    data = data / 255

    # Reverse the power curve
    data = np.power(data, 1 / power)

    # Rescale to max value
    data = data * max_value

    return data


def exif_from_image(pil_image: Image.Image) -> T.Dict[str, T.Any]:
    """
    Get the EXIF data from a PIL image as a dict.
    """
    exif = pil_image.getexif()

    if exif is None or len(exif) == 0:
        return {}

    return {SpectrogramParams.ExifTags(key).name: val for key, val in exif.items()}

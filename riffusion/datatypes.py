"""
Data model for the riffusion API.
"""

from dataclasses import dataclass
import typing as T


@dataclass(frozen=True)
class PromptInput:
    """
    Parameters for one end of interpolation.
    """

    # Text prompt fed into a CLIP model
    prompt: str

    # Random seed for denoising
    seed: int

    # Denoising strength
    denoising: float = 0.75

    # Classifier-free guidance strength
    guidance: float = 7.0


@dataclass(frozen=True)
class InferenceInput:
    """
    Parameters for a single run of the riffusion model, interpolating between
    a start and end set of PromptInputs. This is the API required for a request
    to the model server.
    """

    # Start point of interpolation
    start: PromptInput

    # End point of interpolation
    end: PromptInput

    # Interpolation alpha [0, 1]. A value of 0 uses start fully, a value of 1
    # uses end fully.
    alpha: float

    # Number of inner loops of the diffusion model
    num_inference_steps: int = 50

    # Which seed image to use
    seed_image_id: str = "og_beat"

    # ID of mask image to use
    mask_image_id: T.Optional[str] = None


@dataclass(frozen=True)
class InferenceOutput:
    """
    Response from the model inference server.
    """
    # base64 encoded spectrogram image as a JPEG
    image: str

    # base64 encoded audio clip as an MP3
    audio: str

    # The duration of the audio clip
    duration_s: float

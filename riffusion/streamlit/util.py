"""
Streamlit utilities (mostly cached wrappers around riffusion code).
"""
import io
import threading
import typing as T

import pydub
import streamlit as st
import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

from riffusion.audio_splitter import AudioSplitter
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

# TODO(hayk): Add URL params

DEFAULT_CHECKPOINT = "riffusion/riffusion-model-v1"

AUDIO_EXTENSIONS = ["mp3", "wav", "flac", "webm", "m4a", "ogg"]
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg"]

SCHEDULER_OPTIONS = [
    "DPMSolverMultistepScheduler",
    "PNDMScheduler",
    "DDIMScheduler",
    "LMSDiscreteScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
]


@st.cache_resource
def load_riffusion_checkpoint(
    checkpoint: str = DEFAULT_CHECKPOINT,
    no_traced_unet: bool = False,
    device: str = "cuda",
) -> RiffusionPipeline:
    """
    Load the riffusion pipeline.
    """
    return RiffusionPipeline.load_checkpoint(
        checkpoint=checkpoint,
        use_traced_unet=not no_traced_unet,
        device=device,
    )


@st.cache_resource
def load_stable_diffusion_pipeline(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    scheduler: str = SCHEDULER_OPTIONS[0],
) -> StableDiffusionPipeline:
    """
    Load the riffusion pipeline.

    TODO(hayk): Merge this into RiffusionPipeline to just load one model.
    """
    if device == "cpu" or device.lower().startswith("mps"):
        print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
        dtype = torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=dtype,
        safety_checker=lambda images, **kwargs: (images, False),
    ).to(device)

    pipeline.scheduler = get_scheduler(scheduler, config=pipeline.scheduler.config)

    return pipeline


def get_scheduler(scheduler: str, config: T.Any) -> T.Any:
    """
    Construct a denoising scheduler from a string.
    """
    if scheduler == "PNDMScheduler":
        from diffusers import PNDMScheduler

        return PNDMScheduler.from_config(config)
    elif scheduler == "DPMSolverMultistepScheduler":
        from diffusers import DPMSolverMultistepScheduler

        return DPMSolverMultistepScheduler.from_config(config)
    elif scheduler == "DDIMScheduler":
        from diffusers import DDIMScheduler

        return DDIMScheduler.from_config(config)
    elif scheduler == "LMSDiscreteScheduler":
        from diffusers import LMSDiscreteScheduler

        return LMSDiscreteScheduler.from_config(config)
    elif scheduler == "EulerDiscreteScheduler":
        from diffusers import EulerDiscreteScheduler

        return EulerDiscreteScheduler.from_config(config)
    elif scheduler == "EulerAncestralDiscreteScheduler":
        from diffusers import EulerAncestralDiscreteScheduler

        return EulerAncestralDiscreteScheduler.from_config(config)
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")


@st.cache_resource
def pipeline_lock() -> threading.Lock:
    """
    Singleton lock used to prevent concurrent access to any model pipeline.
    """
    return threading.Lock()


@st.cache_resource
def load_stable_diffusion_img2img_pipeline(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    scheduler: str = SCHEDULER_OPTIONS[0],
) -> StableDiffusionImg2ImgPipeline:
    """
    Load the image to image pipeline.

    TODO(hayk): Merge this into RiffusionPipeline to just load one model.
    """
    if device == "cpu" or device.lower().startswith("mps"):
        print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
        dtype = torch.float32

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=dtype,
        safety_checker=lambda images, **kwargs: (images, False),
    ).to(device)

    pipeline.scheduler = get_scheduler(scheduler, config=pipeline.scheduler.config)

    return pipeline


@st.cache_data(persist=True)
def run_txt2img(
    prompt: str,
    num_inference_steps: int,
    guidance: float,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
) -> Image.Image:
    """
    Run the text to image pipeline with caching.
    """
    with pipeline_lock():
        pipeline = load_stable_diffusion_pipeline(
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )

        generator_device = "cpu" if device.lower().startswith("mps") else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        output = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt or None,
            generator=generator,
            width=width,
            height=height,
        )

        return output["images"][0]


@st.cache_resource
def spectrogram_image_converter(
    params: SpectrogramParams,
    device: str = "cuda",
) -> SpectrogramImageConverter:
    return SpectrogramImageConverter(params=params, device=device)


@st.cache
def spectrogram_image_from_audio(
    segment: pydub.AudioSegment,
    params: SpectrogramParams,
    device: str = "cuda",
) -> Image.Image:
    converter = spectrogram_image_converter(params=params, device=device)
    return converter.spectrogram_image_from_audio(segment)


@st.cache_data
def audio_segment_from_spectrogram_image(
    image: Image.Image,
    params: SpectrogramParams,
    device: str = "cuda",
) -> pydub.AudioSegment:
    converter = spectrogram_image_converter(params=params, device=device)
    return converter.audio_from_spectrogram_image(image)


@st.cache_data
def audio_bytes_from_spectrogram_image(
    image: Image.Image,
    params: SpectrogramParams,
    device: str = "cuda",
    output_format: str = "mp3",
) -> io.BytesIO:
    segment = audio_segment_from_spectrogram_image(image=image, params=params, device=device)

    audio_bytes = io.BytesIO()
    segment.export(audio_bytes, format=output_format)

    return audio_bytes


def select_device(container: T.Any = st.sidebar) -> str:
    """
    Dropdown to select a torch device, with an intelligent default.
    """
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"

    device_options = ["cuda", "cpu", "mps"]
    device = st.sidebar.selectbox(
        "Device",
        options=device_options,
        index=device_options.index(default_device),
        help="Which compute device to use. CUDA is recommended.",
    )
    assert device is not None

    return device


def select_audio_extension(container: T.Any = st.sidebar) -> str:
    """
    Dropdown to select an audio extension, with an intelligent default.
    """
    default = "mp3" if pydub.AudioSegment.ffmpeg else "wav"
    extension = container.selectbox(
        "Output format",
        options=AUDIO_EXTENSIONS,
        index=AUDIO_EXTENSIONS.index(default),
    )
    assert extension is not None
    return extension


def select_scheduler(container: T.Any = st.sidebar) -> str:
    """
    Dropdown to select a scheduler.
    """
    scheduler = st.sidebar.selectbox(
        "Scheduler",
        options=SCHEDULER_OPTIONS,
        index=0,
        help="Which diffusion scheduler to use",
    )
    assert scheduler is not None
    return scheduler


def select_checkpoint(container: T.Any = st.sidebar) -> str:
    """
    Provide a custom model checkpoint.
    """
    return container.text_input(
        "Custom Checkpoint",
        value=DEFAULT_CHECKPOINT,
        help="Provide a custom model checkpoint",
    )


@st.cache_data
def load_audio_file(audio_file: io.BytesIO) -> pydub.AudioSegment:
    return pydub.AudioSegment.from_file(audio_file)


@st.cache_resource
def get_audio_splitter(device: str = "cuda"):
    return AudioSplitter(device=device)


@st.cache_resource
def load_magic_mix_pipeline(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
):
    pipeline = DiffusionPipeline.from_pretrained(
        checkpoint,
        custom_pipeline="magic_mix",
    ).to(device)

    pipeline.scheduler = get_scheduler(scheduler, pipeline.scheduler.config)

    return pipeline


@st.cache
def run_img2img_magic_mix(
    prompt: str,
    init_image: Image.Image,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    kmin: float,
    kmax: float,
    mix_factor: float,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
):
    """
    Run the magic mix pipeline for img2img.
    """
    with pipeline_lock():
        pipeline = load_magic_mix_pipeline(
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )

        return pipeline(
            init_image,
            prompt=prompt,
            kmin=kmin,
            kmax=kmax,
            mix_factor=mix_factor,
            seed=seed,
            guidance_scale=guidance_scale,
            steps=num_inference_steps,
        )


@st.cache
def run_img2img(
    prompt: str,
    init_image: Image.Image,
    denoising_strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: T.Optional[str] = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    scheduler: str = SCHEDULER_OPTIONS[0],
    progress_callback: T.Optional[T.Callable[[float], T.Any]] = None,
) -> Image.Image:
    with pipeline_lock():
        pipeline = load_stable_diffusion_img2img_pipeline(
            checkpoint=checkpoint,
            device=device,
            scheduler=scheduler,
        )

        generator_device = "cpu" if device.lower().startswith("mps") else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        num_expected_steps = max(int(num_inference_steps * denoising_strength), 1)

        def callback(step: int, tensor: torch.Tensor, foo: T.Any) -> None:
            if progress_callback is not None:
                progress_callback(step / num_expected_steps)

        result = pipeline(
            prompt=prompt,
            image=init_image,
            strength=denoising_strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt or None,
            num_images_per_prompt=1,
            generator=generator,
            callback=callback,
            callback_steps=1,
        )

        return result.images[0]


class StreamlitCounter:
    """
    Simple counter stored in streamlit session state.
    """

    def __init__(self, key="_counter"):
        self.key = key
        if not st.session_state.get(self.key):
            st.session_state[self.key] = 0

    def increment(self):
        st.session_state[self.key] += 1

    @property
    def value(self):
        return st.session_state[self.key]


def display_and_download_audio(
    segment: pydub.AudioSegment,
    name: str,
    extension: str = "mp3",
) -> None:
    """
    Display the given audio segment and provide a button to download it with
    a proper file name, since st.audio doesn't support that.
    """
    mime_type = f"audio/{extension}"
    audio_bytes = io.BytesIO()
    segment.export(audio_bytes, format=extension)
    st.audio(audio_bytes, format=mime_type)

    st.download_button(
        f"{name}.{extension}",
        data=audio_bytes,
        file_name=f"{name}.{extension}",
        mime=mime_type,
    )

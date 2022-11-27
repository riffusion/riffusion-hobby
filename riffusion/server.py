"""
Inference server for the riffusion project.
"""

import base64
import dataclasses
import logging
import io
import json
from pathlib import Path
import time
import typing as T

import dacite
import flask
from flask_cors import CORS
import PIL
import torch

from .audio import wav_bytes_from_spectrogram_image
from .audio import mp3_bytes_from_wav_bytes
from .datatypes import InferenceInput
from .datatypes import InferenceOutput
from .riffusion_pipeline import RiffusionPipeline

# Flask app with CORS
app = flask.Flask(__name__)
CORS(app)

# Log at the INFO level to both stdout and disk
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.FileHandler("server.log"))

# Global variable for the model pipeline
MODEL = None

# Where built-in seed images are stored
SEED_IMAGES_DIR = Path(Path(__file__).resolve().parent.parent, "seed_images")


def run_app(
    *,
    checkpoint: str,
    host: str = "127.0.0.1",
    port: int = 3000,
    debug: bool = False,
    ssl_certificate: T.Optional[str] = None,
    ssl_key: T.Optional[str] = None,
):
    """
    Run a flask API that serves the given riffusion model checkpoint.
    """
    # Initialize the model
    global MODEL
    MODEL = load_model(checkpoint=checkpoint)

    args = dict(
        debug=debug,
        threaded=False,
        host=host,
        port=port,
    )

    if ssl_certificate:
        assert ssl_key is not None
        args["ssl_context"] = (ssl_certificate, ssl_key)

    app.run(**args)


def load_model(checkpoint: str):
    """
    Load the riffusion model pipeline.
    """
    assert torch.cuda.is_available()

    model = RiffusionPipeline.from_pretrained(
        checkpoint,
        revision="fp16",
        torch_dtype=torch.float16,
        # Disable the NSFW filter, causes incorrect false positives
        safety_checker=lambda images, **kwargs: (images, False),
    )

    model = model.to("cuda")

    return model


@app.route("/run_inference/", methods=["POST"])
def run_inference():
    """
    Execute the riffusion model as an API.

    Inputs:
        Serialized JSON of the InferenceInput dataclass

    Returns:
        Serialized JSON of the InferenceOutput dataclass
    """
    start_time = time.time()

    # Parse the payload as JSON
    json_data = json.loads(flask.request.data)

    # Log the request
    logging.info(json_data)

    # Parse an InferenceInput dataclass from the payload
    try:
        inputs = dacite.from_dict(InferenceInput, json_data)
    except dacite.exceptions.WrongTypeError as exception:
        logging.info(json_data)
        return str(exception), 400
    except dacite.exceptions.MissingValueError as exception:
        logging.info(json_data)
        return str(exception), 400

    # Load the seed image by ID
    init_image_path = Path(SEED_IMAGES_DIR, f"{inputs.seed_image_id}.png")
    if not init_image_path.is_file:
        return f"Invalid seed image: {inputs.seed_image_id}", 400
    init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

    # Load the mask image by ID
    if inputs.mask_image_id:
        mask_image_path = Path(SEED_IMAGES_DIR, f"{inputs.mask_image_id}.png")
        if not mask_image_path.is_file:
            return f"Invalid mask image: {inputs.mask_image_id}", 400
        mask_image = PIL.Image.open(str(mask_image_path)).convert("RGB")
    else:
        mask_image = None

    # Execute the model to get the spectrogram image
    image = MODEL.riffuse(inputs, init_image=init_image, mask_image=mask_image)

    # Reconstruct audio from the image
    wav_bytes, duration_s = wav_bytes_from_spectrogram_image(image)
    mp3_bytes = mp3_bytes_from_wav_bytes(wav_bytes)

    # Compute the output as base64 encoded strings
    image_bytes = image_bytes_from_image(image, mode="JPEG")

    # Assemble the output dataclass
    output = InferenceOutput(
        image="data:image/jpeg;base64," + base64_encode(image_bytes),
        audio="data:audio/mpeg;base64," + base64_encode(mp3_bytes),
        duration_s=duration_s,
    )

    # Log the total time
    logging.info(f"Request took {time.time() - start_time:.2f} s")

    return flask.jsonify(dataclasses.asdict(output))


def image_bytes_from_image(image: PIL.Image, mode: str = "PNG") -> io.BytesIO:
    """
    Convert a PIL image into bytes of the given image format.
    """
    image_bytes = io.BytesIO()
    image.save(image_bytes, mode)
    image_bytes.seek(0)
    return image_bytes


def base64_encode(buffer: io.BytesIO) -> str:
    """
    Encode the given buffer as base64.
    """
    return base64.encodebytes(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    import argh

    argh.dispatch_command(run_app)

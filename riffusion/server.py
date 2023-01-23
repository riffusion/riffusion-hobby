"""
Flask server that serves the riffusion model as an API.
"""

import dataclasses
import io
import json
import logging
import time
import typing as T
from pathlib import Path

import argh
import dacite
import flask
import PIL
from flask_cors import CORS

from riffusion.datatypes import InferenceInput, InferenceOutput
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import base64_util

# Flask app with CORS
app = flask.Flask(__name__)
CORS(app)

# Log at the INFO level to both stdout and disk
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.FileHandler("server.log"))

# Global variable for the model pipeline
PIPELINE: T.Optional[RiffusionPipeline] = None

# Where built-in seed images are stored
SEED_IMAGES_DIR = Path(Path(__file__).resolve().parent.parent, "seed_images")


def run_app(
    *,
    checkpoint: str = "riffusion/riffusion-model-v1",
    no_traced_unet: bool = False,
    device: str = "cuda",
    host: str = "127.0.0.1",
    port: int = 3013,
    debug: bool = False,
    ssl_certificate: T.Optional[str] = None,
    ssl_key: T.Optional[str] = None,
):
    """
    Run a flask API that serves the given riffusion model checkpoint.
    """
    # Initialize the model
    global PIPELINE
    PIPELINE = RiffusionPipeline.load_checkpoint(
        checkpoint=checkpoint,
        use_traced_unet=not no_traced_unet,
        device=device,
    )

    args = dict(
        debug=debug,
        threaded=False,
        host=host,
        port=port,
    )

    if ssl_certificate:
        assert ssl_key is not None
        args["ssl_context"] = (ssl_certificate, ssl_key)

    app.run(**args)  # type: ignore


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

    response = compute_request(
        inputs=inputs,
        seed_images_dir=SEED_IMAGES_DIR,
        pipeline=PIPELINE,
    )

    # Log the total time
    logging.info(f"Request took {time.time() - start_time:.2f} s")

    return response


def compute_request(
    inputs: InferenceInput,
    pipeline: RiffusionPipeline,
    seed_images_dir: str,
) -> T.Union[str, T.Tuple[str, int]]:
    """
    Does all the heavy lifting of the request.

    Args:
        inputs: The input dataclass
        pipeline: The riffusion model pipeline
        seed_images_dir: The directory where seed images are stored
    """
    # Load the seed image by ID
    init_image_path = Path(seed_images_dir, f"{inputs.seed_image_id}.png")

    if not init_image_path.is_file():
        return f"Invalid seed image: {inputs.seed_image_id}", 400
    init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

    # Load the mask image by ID
    mask_image: T.Optional[PIL.Image.Image] = None
    if inputs.mask_image_id:
        mask_image_path = Path(seed_images_dir, f"{inputs.mask_image_id}.png")
        if not mask_image_path.is_file():
            return f"Invalid mask image: {inputs.mask_image_id}", 400
        mask_image = PIL.Image.open(str(mask_image_path)).convert("RGB")

    # Execute the model to get the spectrogram image
    image = pipeline.riffuse(
        inputs,
        init_image=init_image,
        mask_image=mask_image,
    )

    # TODO(hayk): Change the frequency range to [20, 20k] once the model is retrained
    params = SpectrogramParams(
        min_frequency=0,
        max_frequency=10000,
    )

    # Reconstruct audio from the image
    # TODO(hayk): It may help performance a bit to cache this object
    converter = SpectrogramImageConverter(params=params, device=str(pipeline.device))

    segment = converter.audio_from_spectrogram_image(
        image,
        apply_filters=True,
    )

    # Export audio to MP3 bytes
    mp3_bytes = io.BytesIO()
    segment.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)

    # Export image to JPEG bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, exif=image.getexif(), format="JPEG")
    image_bytes.seek(0)

    # Assemble the output dataclass
    output = InferenceOutput(
        image="data:image/jpeg;base64," + base64_util.encode(image_bytes),
        audio="data:audio/mpeg;base64," + base64_util.encode(mp3_bytes),
        duration_s=segment.duration_seconds,
    )

    return json.dumps(dataclasses.asdict(output))


def main():
    """
    Run the flask app
    """
    argh.dispatch_command(run_app)


if __name__ == "__main__":
    main()

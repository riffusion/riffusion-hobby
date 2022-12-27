# Riffusion

Riffusion is a technique for real-time music and audio generation with stable diffusion.

Read about it at https://www.riffusion.com/about and try it at https://www.riffusion.com/.

* Inference server: https://github.com/riffusion/riffusion
* Web app: https://github.com/riffusion/riffusion-app
* Model checkpoint: https://huggingface.co/riffusion/riffusion-model-v1

This repository contains the Python backend does the model inference and audio processing, including:

 * a diffusers pipeline that performs prompt interpolation combined with image conditioning
 * a module for (approximately) converting between spectrograms and waveforms
 * a flask server to provide model inference via API to the next.js app
 * a model template titled baseten.py for deploying as a Truss


## Install

Tested with Python 3.9 and diffusers 0.9.0.

To run this model, you need a GPU with CUDA. To run it in real time, it needs to be able to run stable diffusion with approximately 50 steps in under five seconds.

You need to make sure you have torch and torchaudio installed with CUDA support. See the [install guide](https://pytorch.org/get-started/locally/) or [stable wheels](https://download.pytorch.org/whl/torch_stable.html).

```
conda create --name riffusion-inference python=3.9
conda activate riffusion-inference
python -m pip install -r requirements.txt
```

If torchaudio has no audio backend, see [this issue](https://github.com/riffusion/riffusion/issues/12).

You can open and save WAV files with pure python. For opening and saving non-wav files – like mp3 – you'll need ffmpeg or libav.

Guides:
* [CUDA help](https://github.com/riffusion/riffusion/issues/3)
* [Windows Simple Instructions](https://www.reddit.com/r/riffusion/comments/zrubc9/installation_guide_for_riffusion_app_inference/)

## Run the model server
Start the Flask server:
```
python -m riffusion.server --host 127.0.0.1 --port 3013
```

You can specify `--checkpoint` with your own directory or huggingface ID in diffusers format.

The model endpoint is now available at `http://127.0.0.1:3013/run_inference` via POST request.

Example input (see [InferenceInput](https://github.com/hmartiro/riffusion-inference/blob/main/riffusion/datatypes.py#L28) for the API):
```
{
  "alpha": 0.75,
  "num_inference_steps": 50,
  "seed_image_id": "og_beat",

  "start": {
    "prompt": "church bells on sunday",
    "seed": 42,
    "denoising": 0.75,
    "guidance": 7.0
  },

  "end": {
    "prompt": "jazz with piano",
    "seed": 123,
    "denoising": 0.75,
    "guidance": 7.0
  }
}
```

Example output (see [InferenceOutput](https://github.com/hmartiro/riffusion-inference/blob/main/riffusion/datatypes.py#L54) for the API):
```
{
  "image": "< base64 encoded JPEG image >",
  "audio": "< base64 encoded MP3 clip >"
}
```

Use the `--device` argument to specify the torch device to use.

`cuda` is recommended.

`cpu` works but is quite slow.

`mps` is supported for inference, but some operations fall back to CPU. You may need to set
PYTORCH_ENABLE_MPS_FALLBACK=1. In addition, it is not deterministic.

## Test
Tests live in the `test/` directory and are implemented with `unittest`.

To run all tests:
```
python -m unittest test/*_test.py
```

To run a single test:
```
python -m unittest test.audio_to_image_test
```

To preserve temporary outputs for debugging, set `RIFFUSION_TEST_DEBUG`:
```
RIFFUSION_TEST_DEBUG=1 python -m unittest test.audio_to_image_test
```

To run a single test case:
```
python -m unittest test.audio_to_image_test -k AudioToImageTest.test_stereo
```

To run tests using a specific torch device, set `RIFFUSION_TEST_DEVICE`. Tests should pass with
`cpu`, `cuda`, and `mps` backends.

## Development
Install additional packages for dev with `pip install -r dev_requirements.txt`.

* Linter: `ruff`
* Formatter: `black`
* Type checker: `mypy`

These are configured in `pyproject.toml`.

The results of `mypy .`, `black .`, and `ruff .` *must* be clean to accept a PR.

## Citation

If you build on this work, please cite it as follows:

```
@article{Forsgren_Martiros_2022,
  author = {Forsgren, Seth* and Martiros, Hayk*},
  title = {{Riffusion - Stable diffusion for real-time music generation}},
  url = {https://riffusion.com/about},
  year = {2022}
}
```

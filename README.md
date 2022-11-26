# Riffusion Inference Server

Model inference backend for the Riffusion app.

 * a diffusers pipeline that performs prompt interpolation combined with image conditioning
 * a module for (approximately) converting between spectrograms and waveforms
 * a flask server to provide model inference via API to the next.js app

## Install
Tested with Python 3.9 and diffusers 0.9.0

```
conda create --name riffusion-inference python=3.9
conda activate riffusion-inference
python -m pip install -r requirements.txt
```

## Run
Start the Flask server:
```
python -m riffusion.server --port 3013 --host 127.0.0.1 --checkpoint /path/to/diffusers_checkpoint
```

The model endpoint is now available at `http://127.0.0.1:3013/run_inference` via POST request.

Example input (see [InferenceInput](https://github.com/hmartiro/riffusion-inference/blob/main/riffusion/datatypes.py#L28) for the API):
```
{
  alpha: 0.75,
  num_inference_steps: 50,
  seed_image_id: 0,

  start: {
    prompt: "church bells on sunday",
    seed: 42,
    denoising: 0.75,
    guidance: 7.0,
  },

  end: {
    prompt: "jazz with piano",
    seed: 123,
    denoising: 0.75,
    guidance: 7.0,
  },
}
```

Example output (see [InferenceOutput](https://github.com/hmartiro/riffusion-inference/blob/main/riffusion/datatypes.py#L54) for the API):
```
{
  image: "< base64 encoded PNG >",
  audio: "< base64 encoded MP3 clip >",,
}
```


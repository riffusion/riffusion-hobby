import json
import typing as T
from pathlib import Path

import streamlit as st

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util

# Example input json file to process in batch
EXAMPLE_INPUT = """
{
  "params": {
    "checkpoint": "riffusion/riffusion-model-v1",
    "scheduler": "DPMSolverMultistepScheduler",
    "num_inference_steps": 50,
    "guidance": 7.0,
    "width": 512,
  },
  "entries": [
    {
      "prompt": "Church bells",
      "seed": 42
    },
    {
      "prompt": "electronic beats",
      "negative_prompt": "drums",
      "seed": 100
    },
    {
      "prompt": "classical violin concerto",
      "seed": 4
    }
  ]
}
"""


def render() -> None:
    st.subheader("📜 Text to Audio Batch")
    st.write(
        """
    Generate audio in batch from a JSON file of text prompts.
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            This tool is a batch form of text_to_audio, where the inputs are read in from a JSON
            file. The input file contains a global params block and a list of entries with positive
            and negative prompts. It's useful for automating a larger set of generations. See the
            example inputs below for the format of the file.
            """
        )

    device = streamlit_util.select_device(st.sidebar)

    # Upload a JSON file
    json_file = st.file_uploader(
        "JSON file",
        type=["json"],
        label_visibility="collapsed",
    )

    # Handle the null case
    if json_file is None:
        st.info("Upload a JSON file containing params and prompts")
        with st.expander("Example inputs.json", expanded=False):
            st.code(EXAMPLE_INPUT)
        return

    # Read in and print it
    data = json.loads(json_file.read())
    with st.expander("Input Data", expanded=False):
        st.json(data)

    # Params can either be a list or a single entry
    if isinstance(data["params"], list):
        param_sets = data["params"]
    else:
        param_sets = [data["params"]]

    entries = data["entries"]

    show_images = st.sidebar.checkbox("Show Images", True)
    num_seeds = st.sidebar.number_input(
        "Num Seeds",
        value=1,
        min_value=1,
        max_value=10,
        help="When > 1, increments the seed and runs multiple for each entry",
    )

    # Optionally specify an output directory
    output_dir = st.sidebar.text_input("Output Directory", "")
    output_path: T.Optional[Path] = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Write title cards for each param set
    title_cols = st.columns(len(param_sets))
    for i, params in enumerate(param_sets):
        col = title_cols[i]

        if "name" not in params:
            params["name"] = f"params[{i}]"

        col.write(f"## Param Set {i}")
        col.json(params)

    for entry_i, entry in enumerate(entries):
        st.write("---")
        print(entry)
        prompt = entry["prompt"]
        negative_prompt = entry.get("negative_prompt", None)

        base_seed = entry.get("seed", 42)

        text = f"##### ({base_seed}) {prompt}"
        if negative_prompt:
            text += f"  \n**Negative**: {negative_prompt}"
        st.write(text)

        for seed in range(base_seed, base_seed + num_seeds):
            cols = st.columns(len(param_sets))
            for i, params in enumerate(param_sets):
                col = cols[i]
                col.write(params["name"])

                image = streamlit_util.run_txt2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    num_inference_steps=params.get("num_inference_steps", 50),
                    guidance=params.get("guidance", 7.0),
                    width=params.get("width", 512),
                    checkpoint=params.get("checkpoint", streamlit_util.DEFAULT_CHECKPOINT),
                    scheduler=params.get("scheduler", streamlit_util.SCHEDULER_OPTIONS[0]),
                    height=512,
                    device=device,
                )

                if show_images:
                    col.image(image)

                # TODO(hayk): Change the frequency range to [20, 20k] once the model is retrained
                p_spectrogram = SpectrogramParams(
                    min_frequency=0,
                    max_frequency=10000,
                )

                output_format = "mp3"
                audio_bytes = streamlit_util.audio_bytes_from_spectrogram_image(
                    image=image,
                    params=p_spectrogram,
                    device=device,
                    output_format=output_format,
                )
                col.audio(audio_bytes)

                if output_path:
                    prompt_slug = entry["prompt"].replace(" ", "_")
                    negative_prompt_slug = entry.get("negative_prompt", "").replace(" ", "_")

                    image_path = (
                        output_path / f"image_{i}_{prompt_slug}_neg_{negative_prompt_slug}.jpg"
                    )
                    image.save(image_path, format="JPEG")
                    entry["image_path"] = str(image_path)

                    audio_path = (
                        output_path
                        / f"audio_{i}_{prompt_slug}_neg_{negative_prompt_slug}.{output_format}"
                    )
                    audio_path.write_bytes(audio_bytes.getbuffer())
                    entry["audio_path"] = str(audio_path)

    if output_path:
        output_json_path = output_path / "index.json"
        output_json_path.write_text(json.dumps(data, indent=4))
        st.info(f"Output written to {str(output_path)}")
    else:
        st.info("Enter output directory in sidebar to save to disk")

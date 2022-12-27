import json
import typing as T
from pathlib import Path

import streamlit as st

from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util


def render_text_to_audio_batch() -> None:
    """
    Render audio from text in batches, reading from a text file.
    """
    json_file = st.file_uploader("JSON file", type=["json"])
    if json_file is None:
        st.info("Upload a JSON file of prompts")
        return

    data = json.loads(json_file.read())

    with st.expander("Data", expanded=False):
        st.json(data)

    params = data["params"]
    entries = data["entries"]

    device = streamlit_util.select_device(st.sidebar)

    show_images = st.sidebar.checkbox("Show Images", True)

    # Optionally specify an output directory
    output_dir = st.sidebar.text_input("Output Directory", "")
    output_path: T.Optional[Path] = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(entries):
        st.write(f"### Entry {i + 1} / {len(entries)}")

        st.write(f"Prompt: {entry['prompt']}")

        negative_prompt = entry.get("negative_prompt", None)
        st.write(f"Negative prompt: {negative_prompt}")

        image = streamlit_util.run_txt2img(
            prompt=entry["prompt"],
            negative_prompt=negative_prompt,
            seed=params["seed"],
            num_inference_steps=params["num_inference_steps"],
            guidance=params["guidance"],
            width=params["width"],
            height=params["height"],
            device=device,
        )

        if show_images:
            st.image(image)

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
        st.audio(audio_bytes)

        if output_path:
            prompt_slug = entry["prompt"].replace(" ", "_")
            negative_prompt_slug = entry.get("negative_prompt", "").replace(" ", "_")

            image_path = output_path / f"image_{i}_{prompt_slug}_neg_{negative_prompt_slug}.jpg"
            image.save(image_path, format="JPEG")
            entry["image_path"] = str(image_path)

            audio_path = (
                output_path / f"audio_{i}_{prompt_slug}_neg_{negative_prompt_slug}.{output_format}"
            )
            audio_path.write_bytes(audio_bytes.getbuffer())
            entry["audio_path"] = str(audio_path)

    if output_path:
        output_json_path = output_path / "index.json"
        output_json_path.write_text(json.dumps(data, indent=4))
        st.info(f"Output written to {str(output_path)}")


if __name__ == "__main__":
    render_text_to_audio_batch()

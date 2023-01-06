import streamlit as st


def render_main():
    st.set_page_config(layout="wide", page_icon="🎸")

    st.title(":guitar: Riffusion Playground")

    left, right = st.columns(2)

    with left:
        create_link(":pencil2: Text to Audio", "/text_to_audio")
        st.write("Generate audio clips from text prompts.")

        create_link(":wave: Audio to Audio", "/audio_to_audio")
        st.write("Upload audio and modify with text prompt.")

        create_link(":performing_arts: Interpolation", "/interpolation")
        st.write("Interpolate between prompts in the latent space.")

        create_link(":scissors: Audio Splitter", "/split_audio")
        st.write("Upload audio and split into vocals, bass, drums, and other.")

    with right:
        create_link(":scroll: Text to Audio Batch", "/text_to_audio_batch")
        st.write("Generate audio in batch from a JSON file of text prompts.")

        create_link(":paperclip: Sample Clips", "/sample_clips")
        st.write("Export short clips from an audio file.")

        create_link(":musical_keyboard: Image to Audio", "/image_to_audio")
        st.write("Reconstruct audio from spectrogram images.")


def create_link(name: str, url: str) -> None:
    st.markdown(
        f"### <a href='{url}' target='_self' style='text-decoration: none;'>{name}</a>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_main()

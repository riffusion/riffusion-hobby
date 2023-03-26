import streamlit as st


def render():
    st.title("✨🎸 Riffusion Playground 🎸✨")

    st.write("Select a task from the sidebar to get started!")

    left, right = st.columns(2)

    with left:
        st.subheader("🌊 Text to Audio")
        st.write("Generate audio clips from text prompts.")

        st.subheader("✨ Audio to Audio")
        st.write("Upload audio and modify with text prompt (interpolation supported).")

        st.subheader("🎭 Interpolation")
        st.write("Interpolate between prompts in the latent space.")

        st.subheader("✂️ Audio Splitter")
        st.write("Split audio into stems like vocals, bass, drums, guitar, etc.")

    with right:
        st.subheader("📜 Text to Audio Batch")
        st.write("Generate audio in batch from a JSON file of text prompts.")

        st.subheader("📎 Sample Clips")
        st.write("Export short clips from an audio file.")

        st.subheader("⏈ Spectrogram to Audio")
        st.write("Reconstruct audio from spectrogram images.")

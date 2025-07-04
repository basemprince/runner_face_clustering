"""
app for runner face clustering using Streamlit.
"""

import shutil
from io import BufferedReader
from pathlib import Path

import streamlit as st

import main

st.title("Runner Face Clustering UI")

debug_mode = st.checkbox("Debug mode", value=False)
extract_bib = st.checkbox("Extract bib number", value=True)
visualize_embeddings = st.checkbox("Visualize embeddings", value=False)
reducer_choice = st.selectbox("Dimensionality reduction", ["None", "pca", "tsne"], index=0)

uploaded_files = st.file_uploader("Upload runner images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Process") and uploaded_files:
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    # Clear previous images
    for existing in images_dir.glob("*"):
        existing.unlink()

    img_paths = []
    for uploaded in uploaded_files:
        img_path = images_dir / uploaded.name
        with open(img_path, "wb") as f:
            f.write(uploaded.getbuffer())
        img_paths.append(str(img_path))

    progress = st.progress(0)

    def update_progress(value):
        """Callback to update progress bar."""
        progress.progress(value)

    summary = main.process_images(
        img_paths,
        debug=debug_mode,
        progress_callback=update_progress,
        extract_bib=extract_bib,
        visualize=visualize_embeddings,
        reduce_method=None if reducer_choice == "None" else reducer_choice,
    )

    for cluster_id, info in summary.items():
        TEXT = f"person#{cluster_id}-bib#{info['bib']}" if info["bib"] else f"person#{cluster_id}"
        folder = Path("output") / (TEXT)
        with st.expander(f"{TEXT}", expanded=False):
            for image_file in folder.glob("*.jpg"):
                st.image(str(image_file))

    if Path("output").exists():
        archive_path = shutil.make_archive("output", "zip", "output")
        archive_file: BufferedReader = open(archive_path, "rb")  # pylint: disable=consider-using-with
        with archive_file:
            st.download_button(
                "Download Results",
                archive_file,
                file_name="output.zip",
                mime="application/zip",
            )
    st.subheader("Runner Summary")
    st.json(summary)

"""
app for runner face clustering using Streamlit.
"""

import shutil
from io import BufferedReader
from itertools import chain
from pathlib import Path

import streamlit as st

import main

image_extensions = ["*.jpg", "*.jpeg", "*.png"]
st.title("Runner Face Clustering UI")

debug_mode = st.checkbox("Debug mode", value=False)
extract_bib = st.checkbox("Extract bib number (very slow)", value=False)
visualize_embeddings = st.checkbox("Visualize embeddings", value=False)
min_face_size = st.number_input("Minimum face pixel height", value=5, min_value=1, max_value=100000)
reducer_choice = st.selectbox("Dimensionality reduction", ["None", "pca", "tsne"], index=1)

uploaded_files = st.file_uploader("Upload runner images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Process") and uploaded_files:
    images_dir = Path("images")
    # Remove everything in images/ to prevent old data contamination
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir()

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
        n_components="auto",
        min_face_size=min_face_size,
    )

    for cluster_id, info in summary.items():
        TEXT = f"person#{cluster_id}-bib#{info['bib']}" if info["bib"] else f"person#{cluster_id}"
        folder = Path("output") / (TEXT)
        with st.expander(TEXT, expanded=False):
            image_files = chain.from_iterable(folder.glob(ext) for ext in image_extensions)
            for image_file in image_files:
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
    # st.subheader("Runner Summary")
    # st.json(summary)

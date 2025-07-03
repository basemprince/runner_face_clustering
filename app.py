from pathlib import Path
import shutil

import streamlit as st

import main

st.title("Runner Face Clustering UI")

debug_mode = st.checkbox("Debug mode", value=False)

uploaded_files = st.file_uploader(
    "Upload runner images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

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
        progress.progress(value)

    summary = main.process_images(
        img_paths, debug=debug_mode, progress_callback=update_progress
    )

    for cluster_id, info in summary.items():
        text = f"bib#{info['bib']}" if info["bib"] else f"person#{cluster_id}"
        folder = Path("output") / (
            text
        )
        with st.expander(f"Cluster {cluster_id} - {text}", expanded=False):
            for image_file in folder.glob("*.jpg"):
                st.image(str(image_file))

    if Path("output").exists():
        archive_path = shutil.make_archive("output", "zip", "output")
        with open(archive_path, "rb") as f:
            st.download_button("Download Results", f, file_name="output.zip", mime="application/zip")

    st.subheader("Runner Summary")
    st.json(summary)

from pathlib import Path

import streamlit as st

import main

st.title("Runner Face Clustering UI")

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

    summary = main.process_images(img_paths, debug=False)

    st.subheader("Runner Summary")
    st.json(summary)
    st.write("Results stored in the output/ directory.")

    for cluster_id, info in summary.items():
        folder = Path("output") / (
            f"bib#{info['bib']}" if info["bib"] else f"person#{cluster_id}"
        )
        st.write(f"## Cluster {cluster_id}")
        for image_file in folder.glob("*.jpg"):
            st.image(str(image_file))

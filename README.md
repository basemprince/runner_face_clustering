# Runner Face Clustering

takes in pictures of runners, recognizes faces, clusters them, identifies bib numbers and saves them into their respective folders

## Running the UI

1. Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate face-cluster-env
```

2. Launch the Streamlit app:

```bash
streamlit run app.py
```

Upload your runner images and press **Process** to generate results in the `output/` directory.
If you encounter a `ModuleNotFoundError` for `cv2`, make sure the environment was created from `environment.yaml` which installs OpenCV.

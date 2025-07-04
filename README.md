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

Each cluster folder contains the original image, the cropped body image, and a
`debug_` version with the face box and OCR result overlaid.

The UI includes **Debug mode** and **Extract bib number** checkboxes to control
whether debug images are generated and whether OCR is run on bib numbers. You
can also enable **Visualize embeddings** to save a 2‑D scatter plot and choose a
dimensionality reducer (`None`, `pca`, or `tsne`).
Cluster results are displayed inside collapsed sections so you can
expand only the clusters you care about. After processing you can download all
output files as a single ZIP archive using the **Download Results** button.
While the images are analyzed, a progress bar shows completion status.

## Command line usage

The processing logic can also run without the UI:

```bash
python main.py --visualize --reduce-method tsne --n-components 2
```

Use `--help` for the full list of options including controlling the reducer and
enabling or disabling OCR.

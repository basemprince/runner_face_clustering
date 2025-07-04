"""
this script processes images to detect runners, extract their faces, and cluster them based on face embeddings.
"""

# pylint: disable=too-many-locals, too-many-branches, cell-var-from-loop,
# pylint: disable=too-many-arguments, too-many-statements,
# pylint: disable=too-many-positional-arguments

import glob
import json
import os
import shutil

import cv2
import numpy as np

from cluster_faces import cluster_face_embeddings
from crop_bodies import crop_person
from detect_bibs import detect_bib_in_crop
from detect_runners import detect_persons
from face_embeddings import extract_face_embeddings
from visualize_embeddings import plot_embeddings, reduce_embeddings

if os.path.exists("output"):
    shutil.rmtree("output")
os.makedirs("output")

DEBUG = True


def apply_clahe(image):
    """Apply CLAHE to enhance contrast in the image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def sharpen(image):
    """Apply sharpening filter to the image."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_image(image):
    """Preprocess the image for better OCR and face detection."""
    image = apply_clahe(image)
    image = sharpen(image)
    return image


def preprocess_for_ocr(image):
    """Preprocess the image for OCR by converting to grayscale and applying adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return binarized


def process_images(
    image_paths,
    debug: bool = True,
    progress_callback=None,
    extract_bib: bool = True,
    visualize: bool = False,
    reduce_method: str | None = None,
    n_components: int | str = 2,
    min_body_size: int = 50,
    min_face_size: int = 20,
):
    """Process a list of image paths to detect runners, extract faces, and cluster them.

    Parameters
    ----------
    image_paths : list[str]
        Paths to input images.
    debug : bool, optional
        Whether to output debug information and images.
    progress_callback : callable | None, optional
        Callback to report progress as a float in ``[0, 1]``.
    extract_bib : bool, optional
        If ``True``, attempt OCR to extract bib numbers from detected runners.
    visualize : bool, optional
        If ``True``, save a 2D plot of embeddings to ``output/embeddings.png``.
    reduce_method : str | None, optional
        Dimensionality reduction method for clustering and visualization.
        Allowed values are ``"pca"`` and ``"tsne"``.
    n_components : int | str, optional
        Number of dimensions for the reducer. If ``"auto"`` with PCA, choose the
        number of components explaining at least 90% variance. Defaults to ``2``.
    min_body_size : int, optional
        Minimum width/height in pixels for detected bodies. Smaller detections are skipped.
    min_face_size : int, optional
        Minimum width/height in pixels for detected faces. Smaller faces are skipped.
    """
    samples = []
    total_images = len(image_paths)

    for idx, img_path in enumerate(image_paths, start=1):
        image = cv2.imread(img_path)
        # image = preprocess_image(image)
        persons = [
            box
            for box in detect_persons(image)
            if (box[2] - box[0] >= min_body_size and box[3] - box[1] >= min_body_size)
        ]

        for box in persons:
            body_crop, _ = crop_person(image, box)
            faces = [
                f
                for f in extract_face_embeddings(body_crop)
                if (f["bbox"][2] - f["bbox"][0] >= min_face_size and f["bbox"][3] - f["bbox"][1] >= min_face_size)
            ]

            if not faces:
                continue

            face = faces[0]  # Assume 1 face per body
            samples.append(
                {
                    "img_path": img_path,
                    "body_crop": body_crop,
                    "face_box": face["bbox"],
                    "embedding": face["embedding"],
                    "person_box": box,
                }
            )

        if progress_callback:
            progress_callback(idx / max(total_images, 1) * 0.5)

    labels = cluster_face_embeddings(
        [s["embedding"] for s in samples],
        reduce_method=reduce_method,
        n_components=n_components,
    )

    if visualize:
        reduced = reduce_embeddings(
            [s["embedding"] for s in samples],
            method=reduce_method or "pca",
            n_components=n_components,
        )
        plot_embeddings(reduced, labels=labels, out_path="output/embeddings.png")

    summary: dict[int, list[dict]] = {}

    for lbl, s in zip(labels, samples):
        summary.setdefault(lbl, []).append(s)

    runner_summary = {}

    total_clusters = len(summary)
    processed_clusters = 0
    for cluster_id, group in summary.items():
        bib_votes: dict[str, int] = {}

        print(f"Processing cluster {cluster_id} with {len(group)} samples")
        for sample in group:
            # ocr_input = preprocess_for_ocr(sample["body_crop"])
            ocr_input = sample["body_crop"]
            if extract_bib:
                bib, non_bib = detect_bib_in_crop(ocr_input, debug=debug)
            else:
                bib, non_bib = None, None

            sample["ocr_text"] = (bib or "") + (non_bib or "")
            if bib:
                bib_votes[bib] = bib_votes.get(bib, 0) + 1

        best_bib = max(bib_votes, key=lambda k: bib_votes[k]) if bib_votes else None
        runner_summary[str(cluster_id)] = {"bib": best_bib, "candidates": bib_votes}

        folder_name = f"person#{cluster_id}-bib#{best_bib}" if best_bib else f"person#{cluster_id}"
        out_dir = os.path.join("output", folder_name)
        os.makedirs(out_dir, exist_ok=True)

        for idx, sample in enumerate(group):
            filename = os.path.basename(sample["img_path"])
            shutil.copy(sample["img_path"], os.path.join(out_dir, f"orig_{idx}_{filename}"))

            if debug:
                debug_img = sample["body_crop"].copy()
                x1, y1, x2, y2 = sample["face_box"]
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    debug_img,
                    f"ID:{cluster_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                if sample.get("ocr_text"):
                    cv2.putText(
                        debug_img,
                        f"BIB:{sample['ocr_text']}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                name, ext = os.path.splitext(filename)
                debug_path = os.path.join(out_dir, f"debug_{idx}_{name}{ext}")
                cv2.imwrite(debug_path, debug_img)

        processed_clusters += 1
        if progress_callback:
            progress_callback(0.5 + processed_clusters / max(total_clusters, 1) * 0.5)

    with open("output/runner_summary.json", "w", encoding="utf-8") as f:
        json.dump(runner_summary, f, indent=2)

    if progress_callback:
        progress_callback(1.0)

    return runner_summary


def main(
    debug: bool = True,
    extract_bib: bool = True,
    visualize: bool = False,
    reduce_method: str | None = None,
    n_components: int | str = 2,
    min_body_size: int = 50,
    min_face_size: int = 20,
):
    """Main function to process images.

    Parameters correspond to :func:`process_images`.
    """
    image_paths = glob.glob("images/*.*")
    process_images(
        image_paths,
        debug=debug,
        extract_bib=extract_bib,
        visualize=visualize,
        reduce_method=reduce_method,
        n_components=n_components,
        min_body_size=min_body_size,
        min_face_size=min_face_size,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Process runner images")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--no-extract-bib",
        dest="extract_bib",
        action="store_false",
        help="Skip OCR for bib extraction",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save a 2D embedding plot to output/embeddings.png",
    )
    parser.add_argument(
        "--reduce-method",
        choices=["pca", "tsne"],
        default=None,
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--n-components",
        default="2",
        help="Number of dimensions for the reducer or 'auto' for PCA",
    )
    parser.add_argument(
        "--min-body-size",
        type=int,
        default=50,
        help="Minimum width/height in pixels for detected bodies",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=20,
        help="Minimum width/height in pixels for detected faces",
    )

    args = parser.parse_args()
    n_components_arg: int | str = "auto" if args.n_components == "auto" else int(args.n_components)
    main(
        debug=args.debug,
        extract_bib=args.extract_bib,
        visualize=args.visualize,
        reduce_method=args.reduce_method,
        n_components=n_components_arg,
        min_body_size=args.min_body_size,
        min_face_size=args.min_face_size,
    )

"""
this script processes images to detect runners, extract their faces, and cluster them based on face embeddings.
"""

# pylint: disable=too-many-locals, too-many-branches, cell-var-from-loop

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


def process_images(image_paths, debug=True, progress_callback=None, extract_bib=True):
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
    """
    samples = []
    total_images = len(image_paths)

    for idx, img_path in enumerate(image_paths, start=1):
        image = cv2.imread(img_path)
        # image = preprocess_image(image)
        persons = detect_persons(image)

        for box in persons:
            body_crop, _ = crop_person(image, box)
            faces = extract_face_embeddings(body_crop)

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

    labels = cluster_face_embeddings([s["embedding"] for s in samples])

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


def main(debug=True, extract_bib=True):
    """Main function to process images."""
    image_paths = glob.glob("images/*.*")
    process_images(image_paths, debug=debug, extract_bib=extract_bib)


if __name__ == "__main__":
    main(DEBUG)

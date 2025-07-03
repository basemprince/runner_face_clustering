import glob
import json
import os
import shutil
import numpy as np
import cv2

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
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def sharpen(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def preprocess_image(image):
    image = apply_clahe(image)
    image = sharpen(image)
    return image

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

    return binarized

def process_images(image_paths, debug=True, progress_callback=None):
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

    summary = {}
    for lbl, s in zip(labels, samples):
        summary.setdefault(lbl, []).append(s)

    runner_summary = {}

    total_clusters = len(summary)
    processed_clusters = 0
    for cluster_id, group in summary.items():
        bib_votes = {}
        print(f"Processing cluster {cluster_id} with {len(group)} samples")
        for sample in group:
            # ocr_input = preprocess_for_ocr(sample["body_crop"])
            ocr_input = sample["body_crop"]
            bib, non_bib = detect_bib_in_crop(ocr_input, debug=debug)
            sample["ocr_text"] = (bib or "") + (non_bib or "")
            if bib:
                bib_votes[bib] = bib_votes.get(bib, 0) + 1

        best_bib = max(bib_votes, key=bib_votes.get) if bib_votes else None
        runner_summary[str(cluster_id)] = {"bib": best_bib, "candidates": bib_votes}

        folder_name = f"bib#{best_bib}" if best_bib else f"person#{cluster_id}"
        out_dir = os.path.join("output", folder_name)
        os.makedirs(out_dir, exist_ok=True)

        for idx, sample in enumerate(group):
            filename = os.path.basename(sample["img_path"])
            shutil.copy(
                sample["img_path"], os.path.join(out_dir, f"orig_{idx}_{filename}")
            )

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

    with open("output/runner_summary.json", "w") as f:
        json.dump(runner_summary, f, indent=2)

    if progress_callback:
        progress_callback(1.0)

    return runner_summary


def main(debug=True):
    image_paths = glob.glob("images/*.*")
    process_images(image_paths, debug=debug)


if __name__ == "__main__":
    main(DEBUG)

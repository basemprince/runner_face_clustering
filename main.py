import argparse
import glob

import cv2

from cluster_faces import cluster_faces, save_faces_grouped_by_id
from detect_bibs import detect_numeric_bibs
from extract_faces import extract_faces_and_embeddings


def main(debug=False):
    image_paths = glob.glob("images/*.jpg") + glob.glob("images/*.png")
    face_entries = []

    print("Step 1: Detecting faces and extracting embeddings...")
    for img_path in image_paths:
        img = cv2.imread(img_path)
        faces = extract_faces_and_embeddings(img, debug=debug)
        for f in faces:
            f["img_path"] = img_path
        face_entries.extend(faces)

    if not face_entries:
        print("No faces found in any image.")
        return

    print("Step 2: Clustering faces...")
    labels = cluster_faces(face_entries)

    print("Step 3: Detecting bib numbers per person...")
    bib_map = {}  # cluster_id → bib number
    for cluster_id in set(labels):
        bib_found = None
        # Go through all images of this person
        for i, (entry, label) in enumerate(zip(face_entries, labels)):
            if label != cluster_id:
                continue
            img = cv2.imread(entry["img_path"])
            bibs = detect_numeric_bibs(img, debug=debug)
            if bibs:
                bib_found = bibs[0]
                break
        bib_map[cluster_id] = bib_found

    print("Step 4: Saving output...")
    save_faces_grouped_by_id(face_entries, labels, bib_map, debug=debug)

    print("✅ Done. Check the `output/` folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Show debug info and plots.")
    args = parser.parse_args()
    main(debug=args.debug)

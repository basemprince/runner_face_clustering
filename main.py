import argparse
import glob

import cv2

from cluster_faces import cluster_faces, save_faces_grouped_by_id
from detect_bibs import detect_numeric_bibs_with_boxes
from extract_faces import extract_faces_and_embeddings


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_distance(b1, b2):
    x1, y1 = bbox_center(b1)
    x2, y2 = bbox_center(b2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def assign_bib_to_cluster(cluster_faces, debug=False):
    bib_votes = {}
    for i, face in enumerate(cluster_faces):
        img = cv2.imread(face["img_path"])
        bibs = detect_numeric_bibs_with_boxes(img, debug=debug)
        if not bibs:
            continue

        # Get all faces from that same image
        image_faces = [f for f in cluster_faces if f["img_path"] == face["img_path"]]
        assignments = match_bibs_to_faces(image_faces, bibs, debug=debug)

        for idx, bib_number in assignments.items():
            assigned_face = image_faces[idx]
            if assigned_face == face:
                bib_votes[bib_number] = bib_votes.get(bib_number, 0) + 1
                break

    if bib_votes:
        return max(bib_votes, key=bib_votes.get)
    return None


def match_bibs_to_faces(faces, bibs, debug=False):
    assignments = {}
    used_faces = set()
    used_bibs = set()

    distances = []
    for face_idx, face in enumerate(faces):
        for bib_idx, (bib_number, bib_box) in enumerate(bibs):
            dist = bbox_distance(face["bbox"], bib_box)
            distances.append((dist, face_idx, bib_idx, bib_number))

    distances.sort()  # Closest matches first

    for dist, face_idx, bib_idx, bib_number in distances:
        if face_idx in used_faces or bib_idx in used_bibs:
            continue
        assignments[face_idx] = bib_number
        used_faces.add(face_idx)
        used_bibs.add(bib_idx)

        if debug:
            print(f"Assigned bib {bib_number} to face {face_idx} (distance: {dist:.2f})")

    return assignments  # dict: face index → bib number


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
        print("No faces found.")
        return

    print("Step 2: Clustering faces...")
    labels = cluster_faces(face_entries)

    print("Step 3: Assigning bibs based on spatial proximity to faces...")
    bib_map = {}
    for cluster_id in set(labels):
        cluster_faces_data = [e for e, lbl in zip(face_entries, labels) if lbl == cluster_id]
        bib_map[cluster_id] = assign_bib_to_cluster(cluster_faces_data, debug=debug)

    print("Step 4: Saving full images per person...")
    save_faces_grouped_by_id(face_entries, labels, bib_map, debug=debug)

    print("✅ Done. Check the `output/` folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Show debug info and save face crops.")
    args = parser.parse_args()
    main(debug=args.debug)

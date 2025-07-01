import os
import cv2
import numpy as np
import json
import shutil
from collections import defaultdict
import hdbscan
from face_embeddings import extract_face_embeddings
from extract_embeddings import extract_reid_embedding
from detect_bibs import detect_bib_in_crop
from detect_runners import detect_persons  # YOLO person detection

DEBUG = True
MIN_AREA_THRESHOLD = 5000  # Skip small detections

def combine_features(body_emb, face_emb, weights=(0.5, 1.0)):
    vecs, total_weight = [], 0.0
    if body_emb is not None:
        body = body_emb / np.linalg.norm(body_emb)
        vecs.append(weights[0] * body)
        total_weight += weights[0]
    if face_emb is not None:
        face = face_emb / np.linalg.norm(face_emb)
        vecs.append(weights[1] * face)
        total_weight += weights[1]
    if not vecs:
        return None  # skip if both missing
    return np.concatenate(vecs) / total_weight

def main(input_dir="all_images", output_dir="output", debug=DEBUG):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    samples = []
    for img_name in sorted(os.listdir(input_dir)):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue
        full_path = os.path.join(input_dir, img_name)
        image = cv2.imread(full_path)
        persons = detect_persons(image)

        for i, box in enumerate(persons):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] * crop.shape[1] < MIN_AREA_THRESHOLD:
                if debug:
                    print(f"⛔ Skipping small crop in {img_name}")
                continue

            body_emb = extract_reid_embedding(crop)
            face_data = extract_face_embeddings(crop)
            face_emb = face_data[0]["embedding"] if face_data else None
            bib, _ = detect_bib_in_crop(crop, debug=debug)

            combined = combine_features(body_emb, face_emb)
            if combined is None or not np.isfinite(combined).all():
                if debug:
                    print(f"⚠️ Skipped: bad or missing embeddings in {img_name}")
                continue

            samples.append({
                "embedding": combined,
                "img_path": full_path,
                "crop": crop,
                "bib": bib,
                "face_box": face_data[0]["bbox"] if face_data else None
            })

    if not samples:
        print("❌ No valid samples found.")
        return

    # clustering
    valid_samples = [s for s in samples if s["embedding"] is not None and isinstance(s["embedding"], np.ndarray) and s["embedding"].ndim == 1 and s["embedding"].shape[0] == 512*2]
    X = np.stack([s["embedding"] for s in valid_samples])

    model = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        metric='euclidean',
        allow_single_cluster=True
    )
    labels = model.fit_predict(X)

    # group by label
    grouped = defaultdict(list)
    for label, sample in zip(labels, samples):
        if label == -1:
            label = 10000 + len(grouped)  # assign unique ID for noise
        grouped[label].append(sample)

    summary = {}
    for cluster_id, group in grouped.items():
        bib_votes = defaultdict(int)
        for sample in group:
            if sample["bib"]:
                bib_votes[sample["bib"]] += 1
        majority_bib = max(bib_votes.items(), key=lambda x: x[1])[0] if bib_votes else None

        folder_name = f"bib_{majority_bib}" if majority_bib else f"cluster_{cluster_id}"
        cluster_dir = os.path.join(output_dir, folder_name)
        os.makedirs(cluster_dir, exist_ok=True)

        for idx, sample in enumerate(group):
            out_path = os.path.join(cluster_dir, f"{idx}_{os.path.basename(sample['img_path'])}")
            cv2.imwrite(out_path, sample["crop"])

            if debug:
                dbg = sample["crop"].copy()
                if sample["face_box"]:
                    fx1, fy1, fx2, fy2 = map(int, sample["face_box"])
                    cv2.rectangle(dbg, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                cv2.putText(dbg, f"Cluster {cluster_id}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if sample["bib"]:
                    cv2.putText(dbg, f"Bib: {sample['bib']}", (10, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
                cv2.imwrite(os.path.join(cluster_dir, f"debug_{idx}.jpg"), dbg)

        summary[str(cluster_id)] = {
            "bib": majority_bib,
            "count": len(group),
            "bib_votes": dict(bib_votes)
        }

    with open(os.path.join(output_dir, "runner_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()

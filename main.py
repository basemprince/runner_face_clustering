import os
import cv2
import glob
import argparse
import matplotlib.pyplot as plt

from detect_bibs import detect_bibs_and_numbers
from extract_faces import extract_faces_and_embeddings
from cluster_faces import cluster_faces, save_faces

def main(debug=False):
    image_paths = glob.glob('images/*.jpg') + glob.glob('images/*.png')
    all_faces = []

    for path in image_paths:
        print(f"Processing {path}")
        img = cv2.imread(path)

        if debug:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.show()

        bibs = detect_bibs_and_numbers(img, debug=debug)
        if debug:
            for bib_text, (x1, y1, x2, y2) in bibs:
                print(f"Bib detected: {bib_text}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Bib Detection")
            plt.show()

        faces = extract_faces_and_embeddings(img, debug=debug)
        all_faces.extend(faces)

    if not all_faces:
        print("No faces found.")
        return

    labels = cluster_faces(all_faces)
    save_faces(all_faces, labels)

    print("Finished. Results saved in /output.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug visualizations.')
    args = parser.parse_args()
    main(debug=args.debug)

import os
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

TRAIN_SPLIT = r"F:\Model_3\annotations\Splits_80_10_10\train_filtered.txt"

YOLO_ROOT = Path(r"F:\Model_3\YOLO")

IMAGE_DIR = YOLO_ROOT / "images" / "train"
LABEL_DIR = YOLO_ROOT / "labels" / "train"

FINAL_IMAGE_DIR = YOLO_ROOT / "images" / "train_final"
FINAL_LABEL_DIR = YOLO_ROOT / "labels" / "train_final"

TARGET_LABEL = "Bp_40Hz"

AUGS_PER_IMAGE = 4

# ------------------------------------------------------------------
# Create output folders
# ------------------------------------------------------------------

FINAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Copy (hard-link) original training set into train_final
# ------------------------------------------------------------------

print("Creating train_final...")

for image in IMAGE_DIR.glob("*.png"):
    dst = FINAL_IMAGE_DIR / image.name
    if not dst.exists():
        os.link(image, dst)

for label in LABEL_DIR.glob("*.txt"):
    dst = FINAL_LABEL_DIR / label.name
    if not dst.exists():
        os.link(label, dst)

print("Original training set linked.")

# ------------------------------------------------------------------
# Find images containing ONLY Bp_40Hz
# ------------------------------------------------------------------

df = pd.read_csv(TRAIN_SPLIT, sep="\t")

image_labels = (
    df.groupby("spectrogram_path")["label"]
      .apply(set)
)

bp40_images = image_labels[image_labels == {TARGET_LABEL}].index.tolist()

print(f"Found {len(bp40_images)} spectrograms containing only {TARGET_LABEL}")

# ------------------------------------------------------------------
# Albumentations pipeline
# ------------------------------------------------------------------

transform = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=1.0,
        ),

        A.GaussNoise(
            std_range=(0.005, 0.02),
            p=1.0,
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
    ),
)

# ------------------------------------------------------------------
# Augment
# ------------------------------------------------------------------

created = 0

for image_path in bp40_images:

    print(F"Augmenting {Path(image_path).name}")
    image_path = Path(image_path)

    image_name = image_path.name

    yolo_image = IMAGE_DIR / image_name
    yolo_label = LABEL_DIR / (image_path.stem + ".txt")

    if not yolo_image.exists():
        print(f"Missing image: {yolo_image}")
        continue

    if not yolo_label.exists():
        print(f"Missing label: {yolo_label}")
        continue

    image = cv2.imread(str(yolo_image))

    if image is None:
        print(f"Could not read {yolo_image}")
        continue

    bboxes = []
    classes = []

    with open(yolo_label) as f:
        for line in f:
            cls, x, y, w, h = map(float, line.split())
            classes.append(int(cls))
            bboxes.append([x, y, w, h])
    eps = 1e-6
    fixed_boxes = []

    for idx, (x, y, w, h) in enumerate(bboxes):
        # Convert YOLO -> corners
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2
        
        # Check whether clipping would be required
        needs_fix = (
            x_min < 0.0 or
            y_min < 0.0 or
            x_max > 1.0 or
            y_max > 1.0
        )

        if needs_fix:
            print(f"\nLabel file: {yolo_label}")
            print(f"Box #{idx}")
            print(f"Original YOLO: x={x:.8f}, y={y:.8f}, w={w:.8f}, h={h:.8f}")
            print(f"Corners: x_min={x_min:.8f}, y_min={y_min:.8f}, "
                f"x_max={x_max:.8f}, y_max={y_max:.8f}")
        
        # Clip to image boundaries
        x_min = max(eps, min(x_min, 1.0 - eps))
        y_min = max(eps, min(y_min, 1.0 - eps))
        x_max = max(eps, min(x_max, 1.0 - eps))
        y_max = max(eps, min(y_max, 1.0 - eps))

        # Convert back to YOLO format
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        fixed_boxes.append([x, y, w, h])

    bboxes = fixed_boxes
    for i in range(AUGS_PER_IMAGE):

        aug = transform(
            image=image,
            bboxes=bboxes,
            class_labels=classes,
        )

        new_image = aug["image"]
        new_boxes = aug["bboxes"]

        new_name = f"{image_path.stem}_aug{i}.png"

        out_image = FINAL_IMAGE_DIR / new_name
        out_label = FINAL_LABEL_DIR / f"{image_path.stem}_aug{i}.txt"

        cv2.imwrite(str(out_image), new_image)

        with open(out_label, "w") as f:
            for cls, box in zip(classes, new_boxes):
                x, y, w, h = box
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        created += 1

print(f"\nCreated {created} augmented spectrograms.")
print(f"Final training dataset: {FINAL_IMAGE_DIR}")
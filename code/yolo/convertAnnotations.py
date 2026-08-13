'''
Author: MN
Date: 8/13/26

Description:
    Converts training, validation, and test annotations into YOLO label format.
    Bounding boxes are normalized to image dimensions and saved as individual
    label files for each spectrogram, including empty files for hard negatives.
'''


import os
import pandas as pd
from PIL import Image

###############################################################
# INPUT FILES
###############################################################

INPUT_FILES = [
    r"F:\Model_3\annotations\Splits_80_10_10\train_filtered.txt",
    r"F:\Model_3\annotations\Splits_80_10_10\validation_filtered.txt",
    r"F:\Model_3\annotations\Splits_80_10_10\test_filtered.txt",
]

OUTPUT_BASE_FOLDER = r"F:\Model_3\YOLO\labels"

###############################################################
# CLASS MAPPING
###############################################################

LABEL_MAPPING = {
    "Bm_A_North_Atlantic": 0,
    "Ba_pulse-call": 1,
    "Bp_20Hz": 2,
    "Bp_40Hz": 3,
    "Bb_down-sweep": 4,
}

###############################################################
# PROCESS EACH SPLIT
###############################################################

for INPUT_FILE in INPUT_FILES:

    split_name = os.path.splitext(os.path.basename(INPUT_FILE))[0].replace("_filtered", "")
    OUTPUT_LABEL_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, split_name)

    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)

    ###########################################################
    # LOAD DATA
    ###########################################################

    ext = os.path.splitext(INPUT_FILE)[1]

    if ext == ".csv":
        data = pd.read_csv(INPUT_FILE)
    else:
        data = pd.read_csv(INPUT_FILE, sep="\t")

    ###########################################################
    # CONVERT EACH IMAGE
    ###########################################################

    grouped = data.groupby("spectrogram_path")

    for image_path, rows in grouped:

        image = Image.open(image_path)
        image_width, image_height = image.size

        label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(OUTPUT_LABEL_FOLDER, label_filename)

        with open(label_path, "w") as f:

            # Hard negative (empty label file)
            if pd.isna(rows.iloc[0]["label"]):
                continue

            for _, row in rows.iterrows():

                cls = LABEL_MAPPING[row["label"]]

                xmin = row["xmin"]
                ymin = row["ymin"]
                xmax = row["xmax"]
                ymax = row["ymax"]

                width = xmax - xmin
                height = ymax - ymin

                x_center = xmin + width / 2
                y_center = ymin + height / 2

                # Normalize
                x_center /= image_width
                y_center /= image_height
                width /= image_width
                height /= image_height

                f.write(
                    f"{cls} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

    print(f"Finished converting {split_name}")

print("All annotation files converted.")
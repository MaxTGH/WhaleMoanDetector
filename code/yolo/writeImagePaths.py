import os
import pandas as pd

TRAIN_SPLIT = r"F:\Model_3\annotations\Splits_80_10_10\train_filtered.txt"
VAL_SPLIT = r"F:\Model_3\annotations\Splits_80_10_10\validation_filtered.txt"
TEST_SPLIT = r"F:\Model_3\annotations\Splits_80_10_10\test_filtered.txt"

YOLO_ROOT = r"F:\Model_3\YOLO"


def make_hardlinks(split_file, split_name):

    df = pd.read_csv(split_file, sep="\t")

    images = sorted(df["spectrogram_path"].unique())

    print(f"{split_name}:")
    print(f"  Rows in split: {len(df)}")
    print(f"  Unique spectrograms: {len(images)}")

    image_output = os.path.join(
        YOLO_ROOT,
        "images",
        split_name
    )

    os.makedirs(image_output, exist_ok=True)

    print(f"Creating {len(images)} hard links...")

    for image in images:

        
        filename = os.path.basename(image)
        label_path = os.path.join(
            YOLO_ROOT,
            "labels",
            split_name,
            os.path.splitext(filename)[0] + ".txt"
        )

        if not os.path.exists(label_path):
            print(f"Warning: missing label for {filename}")
            continue
        
        destination = os.path.join(
            image_output,
            filename
        )

        if os.path.exists(destination):
            continue

        os.link(image, destination)

    print(f"{split_name} complete.")


make_hardlinks(TRAIN_SPLIT, "train")
make_hardlinks(VAL_SPLIT, "validation")
make_hardlinks(TEST_SPLIT, "test")
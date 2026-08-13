'''
Author: MN
Date: 8/13/26

Description:
    Evaluates the trained YOLO model on the test dataset.
'''

from ultralytics import YOLO

def main():
    model = YOLO(
        r"F:\Tools\WhaleMoanDetectorGit\code\yolo\runs\detect\train-4\weights\best.pt"
    )

    metrics = model.val(
        data=r"F:\Tools\WhaleMoanDetectorGit\code\yolo\dataset.yaml",
        split="test"
    )

if __name__ == "__main__":
    main()
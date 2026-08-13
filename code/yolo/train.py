from ultralytics import YOLO
import yaml
from pathlib import Path


def main():

    # Load pretrained YOLO11 medium model
    # model = YOLO("yolo11m.pt")

    # Load not pretrained YOLO11 medium model
    

    PROJECT_DIR = Path(__file__).resolve().parent.parent

    CONFIG_PATH = PROJECT_DIR / "config.yaml"
    
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    model_name = config["train_yolo"]["model_name"]

    model = YOLO("yolo11l.pt")

    model.train(
        data=r"F:\Tools\WhaleMoanDetectorGit\code\yolo\dataset.yaml",
        epochs=100,
        #imgsz=1024, # pads the top and bottom of the image
        imgsz=640,
        batch=4,
        patience=20,
        workers=4,
        optimizer="SGD",
        lr0=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        #mosaic=1.0,
        #close_mosaic=20,
        name=model_name
    )


if __name__ == "__main__":
    main()
from ultralytics import YOLO
import os
import yaml
from tqdm import tqdm
from pathlib import Path
import sys
import torch


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from call_context_filter import *
from inference_functions import *

CONFIG_PATH = PROJECT_DIR / "config.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

wav_folder = config["inference_yolo"]["wav_folder"]

detections_folder = config["inference_yolo"]["detections_folder"]
os.makedirs(detections_folder, exist_ok=True)

model_path = os.path.join(
    config["inference_yolo"]["model_path"]
)

model = YOLO(model_path)

fieldnames = [
    "wav_file_path",
    "model_no",
    "image_file_path",
    "label",
    "score",
    "start_time_sec",
    "end_time_sec",
    "start_time",
    "end_time",
    "min_frequency",
    "max_frequency",
    "box_x1",
    "box_x2",
    "box_y1",
    "box_y2",
]

detections_path = os.path.join(detections_folder, "raw_detections_yolo.txt")

with open(detections_path, "w") as f:
    f.write("\t".join(fieldnames) + "\n")

for root, _, files in os.walk(wav_folder):

    wav_files = [f for f in files if f.lower().endswith((".wav", ".x.wav"))]

    for wav_file in tqdm(wav_files):

        wav_path = os.path.join(root, wav_file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detections = predict_and_save_spectrograms_yolo(
            wav_path,
            model,
            config["train"]["model_name"],
            device
        )

        with open(detections_path, "a") as f:
            for det in detections:
                f.write("\t".join(str(det[k]) for k in fieldnames) + "\n")

# call_context_filter(detections_path)
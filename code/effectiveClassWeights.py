import pandas as pd
import torch
import yaml

# Read config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load settings from config
categories = config["categories"]
labeled_data_folder = config["train"]["labeled_data_folder"]
train_set_file = config["train"]["train_set_file"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read training annotations
train_df = pd.read_csv(
    f"{labeled_data_folder}/{train_set_file}",
    sep="\t"
)

counts = []

print("\nTraining set class counts:")

# Compute counts in the exact order defined by config.yaml
for class_name, class_id in categories.items():
    count = (train_df["label"] == class_name).sum()
    counts.append(count)
    print(f"{class_name:25s}: {count}")

counts = torch.tensor(counts, dtype=torch.float32)

# Class-Balanced Loss weights (Cui et al. 2019)
beta = 0.999

effective_num = 1.0 - torch.pow(beta, counts)

weights = (1.0 - beta) / effective_num

# Normalize so the average foreground weight is 1
weights = weights / weights.mean()

# Add background class (label 0)
background_weight = torch.tensor([1.0])

class_weights = torch.cat([
    background_weight,
    weights
]).to(device)

print("\nClass weights:")
print(class_weights)
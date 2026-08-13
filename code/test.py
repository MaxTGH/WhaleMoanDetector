"""
Created on July 2nd 2025

@author: Shane Andres

Runs inference and generates evaluation metrics for any dataset

"""


import torch
from torch.utils.data import DataLoader
from AudioDetectionDataset import AudioDetectionData_with_hard_negatives
from custom_collate import custom_collate
from validation import validation
#from validation_yolo import validation_yolo
import os
import yaml
from model_functions import *



# loading file paths

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
model_folder = config['train']['model_folder']
labeled_data_folder = config['train']['labeled_data_folder']
evaluation_folder = config['train']['evaluation_folder']
categories = config['categories']
model_name = config['test']['model_name']
model_constructor = config['test']['model_constructor']
val_set_file = config['test']['test_set_file']
model = config['test']['model']
iou_threshold = config['test']['iou_threshold']

MODEL_CONSTRUCTORS = {
    "RCNN_ResNet_50": RCNN_ResNet_50,
    "RCNN_COCO": RCNN_COCO,
    "RCNN_WMD_BEST": RCNN_WMD_BEST,
    "RetinaNet_COCO": RetinaNet_COCO,
    "RetinaNet_Scratch": RetinaNet_Scratch
}

model_constructor = MODEL_CONSTRUCTORS[model_constructor]

num_classes = len(categories) + 1 # Five classes plus background


# model architecture

validation_log_folder = f'{evaluation_folder}/{model_name}'

model_path = os.path.join(
    model_folder,
    model_name,
    model
)

model = model_constructor(num_classes)

checkpoint = torch.load(model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
val_batch_size = 1


# loading data

val_set_name = os.path.splitext(val_set_file)[0]
val_loader = DataLoader(AudioDetectionData_with_hard_negatives(csv_file=f'{labeled_data_folder}/{val_set_file}'),
                      batch_size=val_batch_size,
                      shuffle = False,
                      collate_fn = custom_collate,
                      pin_memory = True if torch.cuda.is_available() else False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if str(device) != "cuda":
    print("Using " + str(device))


# running eval

model.to(device)
model.eval()
with torch.no_grad():
    PR_header = (
        f"\n========== Inference on {model_name} ==========\n"
        f'Pulling labels from: {val_set_file}\n'
        f'Using an IOU threshold of: {iou_threshold}\n')
    #PR_output, AP_list = validation_yolo(val_loader, device, model, categories, iou_threshold=iou_threshold, save_dir=validation_log_folder)
    PR_output, AP_list = validation(val_loader, device, model, categories, iou_threshold=iou_threshold)
    PR_file_path = f"{validation_log_folder}/{val_set_name}_{model_name}_{int(100*iou_threshold)}_percent_iou_v2.txt"
    with open(PR_file_path, "w") as f:
        f.write(PR_output)

    print(f"\n========== Inference on {model_name}: ==========")
    print(f'Pulling labels from: {val_set_file}')
    print(f'Using an IOU threshold of: {iou_threshold}')
    for category_name in categories:
        print(f"{category_name} AP: {AP_list[category_name]}")
    


    

 
    
    
    
        
        
        
        
        
        
        
        

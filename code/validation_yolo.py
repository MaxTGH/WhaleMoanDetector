"""
validation_yolo.py

YOLO-style validation.

Computes:
    - Precision-Recall curves
    - Average Precision (AP)

Unlike validation.py, this evaluates every prediction rather than
fixed confidence thresholds.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml

from torchvision.ops import box_iou
from sklearn.metrics import auc


def validation_yolo(
    val_loader,
    device,
    model,
    categories,
    iou_threshold=0.5,
    save_dir="."
):

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    nms_threshold = config["inference"]["nms_threshold"]

    os.makedirs(save_dir, exist_ok=True)

    class_predictions = {
        class_id: {
            "scores": [],
            "tp": [],
            "n_gt": 0
        }
        for class_id in categories.values()
    }

    ####################################################################
    # Run validation
    ####################################################################

    for batch in val_loader:

        for data in batch:

            img = data[0].to(device)

            gt_boxes = (
                data[1]["boxes"].to(device)
                if data[1] and data[1]["boxes"] is not None
                else torch.empty((0, 4), device=device)
            )

            gt_labels = (
                data[1]["labels"].to(device)
                if data[1] and data[1]["labels"] is not None
                else torch.empty((0,), dtype=torch.long, device=device)
            )

            ##############################################################

            output = model([img])[0]

            keep = torchvision.ops.nms(
                output["boxes"],
                output["scores"],
                nms_threshold
            )

            pred_boxes = output["boxes"][keep]
            pred_scores = output["scores"][keep]
            pred_labels = output["labels"][keep]

            order = torch.argsort(pred_scores, descending=True)

            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            pred_labels = pred_labels[order]

            ##############################################################
            # count GT objects
            ##############################################################

            for cls in gt_labels:
                class_predictions[int(cls)]["n_gt"] += 1

            ##############################################################
            # Match predictions
            ##############################################################

            matched_gt = torch.zeros(
                len(gt_boxes),
                dtype=torch.bool,
                device=device,
            )

            if len(gt_boxes):

                ious = box_iou(pred_boxes, gt_boxes)

            else:

                ious = None

            ##############################################################

            for i in range(len(pred_boxes)):

                cls = int(pred_labels[i])

                score = float(pred_scores[i])

                class_predictions[cls]["scores"].append(score)

                tp = 0

                if len(gt_boxes):

                    max_iou, gt_idx = ious[i].max(0)

                    if (
                        max_iou >= iou_threshold
                        and gt_labels[gt_idx] == pred_labels[i]
                        and not matched_gt[gt_idx]
                    ):
                        tp = 1
                        matched_gt[gt_idx] = True

                class_predictions[cls]["tp"].append(tp)

    ####################################################################
    # Compute PR curves
    ####################################################################

    AP = {}

    output_string = ""

    for class_name, class_id in categories.items():

        scores = np.array(class_predictions[class_id]["scores"])
        tp = np.array(class_predictions[class_id]["tp"])

        n_gt = class_predictions[class_id]["n_gt"]

        if len(scores) == 0:

            AP[class_name] = 0

            continue

        ##############################################################

        order = np.argsort(-scores)

        scores = scores[order]
        tp = tp[order]

        fp = 1 - tp

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        precision = tp_cum / (tp_cum + fp_cum + 1e-16)

        recall = tp_cum / max(n_gt, 1)

        ##############################################################
        # Interpolate like YOLO
        ##############################################################

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        AP[class_name] = auc(mrec, mpre)

        ##############################################################
        # Save PR curve
        ##############################################################

        plt.figure(figsize=(6, 6))

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"AP={AP[class_name]:.3f}"
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.title(class_name)

        plt.xlim(0, 1)

        plt.ylim(0, 1)

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{class_name}_PR_curve.png"
            ),
            dpi=300,
        )

        plt.close()

        output_string += (
            f"{class_name:15s}"
            f" AP: {AP[class_name]:.4f}\n"
        )

    return output_string, AP
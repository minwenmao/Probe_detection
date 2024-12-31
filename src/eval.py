import sys
import os
import random
import numpy as np
import cv2
import pandas as pd
import ast
from utils import draw_detections, compute_iou, xywh2xyxy
from ultralytics import YOLO

# Load the ONNX model and ground truth data
onnx_model = YOLO("output_colab/weights/best.onnx")
info_indexed = pd.read_csv('data/info_indexed.csv').set_index('file_name')
info_indexed['bbox'] = info_indexed['bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Prepare validation image paths
folder_path = "data/yolov11/images/val/"
all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

# Process each image
for i in all_images:
    result = onnx_model(i)[0]  # Inference on the image
    boxes = result.boxes
    cv_img = draw_detections(result.orig_img, boxes, boxes.conf)  # Draw detections on image

    ##### Comment below if no ground truth
    gt = info_indexed.loc[i.split('/')[-1]]['bbox']  # Ground truth bbox
    box_true = xywh2xyxy(gt)
    x1, y1, x2, y2 = map(int, box_true[:4])
    cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw ground truth bbox


    if boxes.xyxy.size(0) == 0:  # Handle no detections
        cv2.putText(cv_img, "No detections found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        with open('iou.csv', 'a') as f:
            f.write(f"{i.split('/')[-1]},0\n")
    else:
        iou = compute_iou(box_true, boxes.xyxy)[0]  # Calculate IoU
        with open('iou.csv', 'a') as f:
            f.write(f"{i.split('/')[-1]},{iou.item()}\n")
    ##### Comment above if no ground truth

    ##### Uncomment below if no ground truth
    # if boxes.xyxy.size(0) == 0:  # Handle no detections
    #     cv2.putText(cv_img, "No detections found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     # Other notification
    ###### Uncomment above if no ground truth
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Display the image
    cv2.imshow('output', cv_img)
    cv2.waitKey(0)

# Calculate and print average IoU
avg_iou = pd.read_csv('iou.csv', names=['name', 'iou'])['iou'].mean()
print('The average IoU for the set is:', avg_iou)

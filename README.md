# Probe_detection
## Project Overview
This project focuses on probe detection using the YOLO model, including data preprocessing, model inference, and performance evaluation.
```
data/
│
├── yolov11/
│   ├── images/       # Contains training and validation datasets
│   │   ├── train/    # Training images
│   │   └── val/      # Validation images
│   ├── labels/       # YOLO-format label files
│       ├── train/    # Training set labels
│       └── val/      # Validation set labels
│
├── output_colab/     # Output from model training
│   ├── weights/
│       ├── best.onnx # Best-performing model weights (ONNX format， derived from best.pt to speed up inference)
│       ├── best.pt   # best model trained
│       └── last.pt   # last model trained
│
├── scripts/          # Python scripts for project tasks
│   ├── train.py      # Script for training the model
│   ├── eval.py       # Script for evaluating the model
│   └── utils.py      # Utility functions
│
├── data/info_indexed.csv # Indexed dataset information (e.g., bounding boxes)
├── iou.csv           # IoU computation results
├── README.md         # Project documentation
```

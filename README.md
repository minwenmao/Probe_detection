# Probe_detection
## Project Overview
This project focuses on probe detection using the YOLO model, including data preprocessing, model inference, and performance evaluation.
```
data/
│
├── yolov11/
│   ├── images/         # Contains training and validation datasets
│   │   ├── train/      # Training images
│   │   └── val/        # Validation images
│   ├── labels/         # YOLO-format label files
│       ├── train/      # Training set labels
│       └── val/        # Validation set labels
│
├── output_colab/       # Output from model training
│   ├── weights/
│       ├── best.onnx   # Best-performing model weights (ONNX format， derived from best.pt to speed up inference)
│       ├── best.pt     # best model trained
│       └── last.pt     # last model trained
│   ├── result.csv      # train box loss, val box loss, precision and other infomation of every epoch
│
├── src/          # Python scripts for project tasks
│   ├── prepocessing.py # Script for convert json into dataset used for YOLO
│   ├── eval.py         # Script for evaluating the model with visualization
│   └── utils.py        # Utility functions
│
├── data/info_indexed.csv # Indexed dataset information (e.g., bounding boxes)
├── iou.csv             # IoU computation results
├── README.md           # Project documentation
```

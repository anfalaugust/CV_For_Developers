# CV For Developers — Boat Analytics & Chess Piece Detection with Ultralytics YOLO

An end-to-end computer-vision project built with [Ultralytics](https://docs.ultralytics.com/)
YOLO11. It covers the full pipeline: running core vision tasks (detection and segmentation),
applying a model to a real video-analytics use case, fine-tuning on a custom dataset, evaluating
the trained model, and exporting it for deployment.

## Overview

This project demonstrates two complementary use cases:

1. **Boat & person video analytics** — detection and instance segmentation of boats on images,
   then a real OpenCV video pipeline that counts boats and people inside a defined region of a
   video stream using an Ultralytics `solutions` module.
2. **Chess piece detection** — a YOLO11 model fine-tuned on a custom chess-pieces dataset to
   detect and classify all 13 piece types (both colours), then evaluated and exported for
   deployment.

**Tasks and models used:** object detection (`yolo11n.pt`), instance segmentation
(`yolo11n-seg.pt`), object tracking / region counting (Ultralytics `solutions.RegionCounter`),
custom training (`model.train`), evaluation (`model.val`), and export (`model.export`).

## Repository Structure

```
CV_For_Developers/
├── notebooks/
│   ├── CV_Project1.ipynb    # Detection + segmentation + video region counting
│   └── CV_project2.ipynb    # Custom training + evaluation + export
├── README.md
└── .gitignore
```

## Prerequisites

- Python 3.x
- [ultralytics](https://pypi.org/project/ultralytics/)
- opencv-python
- matplotlib, Pillow, pyyaml

## Model Weights & Dataset

- **Pretrained weights:** `yolo11n.pt` and `yolo11n-seg.pt` — downloaded automatically by
  Ultralytics on first use.
- **Trained weights:** produced by the training run in `CV_project2.ipynb`, saved to
  `runs/detect/chess_runs/chess_yolo11n/weights/best.pt` (not committed — see `.gitignore`).
- **Dataset:** Chess Pieces dataset from
  [Roboflow Universe](https://universe.roboflow.com/) (YOLOv8 format), 13 classes,
  202 train / 58 valid / 29 test images. Download from Roboflow and place the exported zip
  alongside the notebook (see the extraction cell in `CV_project2.ipynb`).

## Installation

```bash
pip install ultralytics opencv-python matplotlib pillow pyyaml
```

## How to Run

The project is delivered as two notebooks — run them top to bottom:

- **`notebooks/CV_Project1.ipynb`** — runs detection + segmentation on boat images and the
  region-counting video pipeline. Provide input images in an `images/` folder and a video at
  `videos/boats.mp4`.
- **`notebooks/CV_project2.ipynb`** — extracts the chess dataset, fine-tunes YOLO11, evaluates
  on the test split, and exports to ONNX.

## Technical Documentation

### Pipeline overview

**Notebook 1 — Core tasks & video analytics**
- Loads `yolo11n.pt` and runs `predict` on boat images (detection).
- Loads `yolo11n-seg.pt` and runs `predict` for instance segmentation (boat masks).
- Builds an OpenCV pipeline (`cv2.VideoCapture` → process → `cv2.VideoWriter`) that samples the
  video to a target FPS and uses `solutions.RegionCounter` to count boats and people inside a
  polygon region.

**Notebook 2 — Training, evaluation, export**
- Extracts the Roboflow chess dataset and fixes the `data.yaml` train/val/test paths.
- Fine-tunes `yolo11n.pt` with `model.train` (60 epochs, imgsz 640, batch 16, `freeze=10`,
  early stopping, and augmentation).
- Evaluates the trained `best.pt` with `model.val` on the test split.
- Exports the model with `model.export(format="onnx")`.

### Training

Fine-tuned `yolo11n.pt` for 60 epochs at image size 640, batch 16, with `patience=15` for early
stopping and `freeze=10` to keep the backbone frozen and train mainly the head — a sensible
choice for a small (~200 image) dataset. Augmentation (rotation, translation, scale, horizontal
flip, mosaic, HSV jitter) was used to add variety and limit overfitting. Training curves are
saved to `results.png`.

### Evaluation

Evaluated on the held-out **test** split at `conf=0.25` and `iou=0.6`, reporting mAP50,
mAP50-95, precision, recall, a confusion matrix, and a PR curve. The 0.25 confidence keeps
recall reasonable so real pieces aren't missed; `iou=0.6` is a moderately strict overlap
requirement, appropriate because pieces are small and sit close together. The confusion matrix
shows a strong diagonal (pawns easiest) with the main errors being false negatives — pieces
occasionally predicted as background — rather than cross-class confusion. For a board-reading
use case, those missed detections are the costly errors, since one undetected piece corrupts
the board state.

### Deployment / Export

The trained `best.pt` is exported to **ONNX** via `model.export(format="onnx")`. ONNX is chosen
because it is a portable, framework-neutral format that runs across many runtimes and hardware
without a PyTorch dependency, making the model easy to serve or embed in a lightweight app.

## Training Program Attribution

Completed under the **Computer Vision for Developers with Ultralytics** program, delivered by
**SDAIA Academy** (5-day on-site capstone, 30 training hours).
Cohort/session: 9–13 August 2026 (Sunday–Thursday).

SDAIA Academy on GitHub: https://github.com/SDAIAAcademy

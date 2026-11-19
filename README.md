# Real-Time Object Detection and Counting Using YOLO
**Case Study:** University Campus CCTV Surveillance  
**Course:** Deep Learning Laboratory — Semester 7, Assignment 8

## 1. Project Overview
This project demonstrates a computer-vision pipeline for detecting and counting people and vehicles in CCTV-style images. It uses the **YOLO11n** object-detection model to locate objects, display bounding boxes, class labels and confidence scores, and count selected categories.

The notebook fine-tunes a pretrained YOLO11n checkpoint on the **COCO8 sample dataset** for 10 epochs, evaluates the model, analyses training losses, and demonstrates detection on a sample CCTV-style image.

> **Dataset note:** The experiment uses COCO8, not a custom university-campus CCTV dataset. The reported metrics are from a very small validation set and should be treated as laboratory results, not campus deployment performance.

## 2. Objectives
- Detect people and selected vehicle categories in images.
- Draw bounding boxes with class labels and confidence scores.
- Count detected objects by category.
- Fine-tune a pretrained YOLO11n model.
- Monitor training and validation losses.
- Evaluate the model using Precision, Recall and mAP.
- Save the trained model checkpoint.

## 3. Technologies Used
- **Python** — implementation
- **Ultralytics YOLO11n** — object detection and training
- **PyTorch** — deep-learning framework
- **OpenCV** — image/video processing
- **NumPy, Pandas** — data handling
- **Matplotlib** — result and loss visualization
- **Jupyter Notebook / Google Colab** — experiment environment

## 4. Methodology
The workflow consists of the following stages:

1. Load the pretrained `yolo11n.pt` model.
2. Load the COCO8 dataset using `coco8.yaml`.
3. Fine-tune the model for 10 epochs with image size 640 and batch size 16.
4. Record training and validation losses.
5. Evaluate the best checkpoint on the validation split.
6. Run prediction on a CCTV-style image.
7. Filter detections for the target classes and calculate class-wise counts.

**Target classes:** `person`, `bicycle`, `motorcycle`, `car`, `bus`, and `truck`.

The counting logic counts detections in an image/frame. It does not track unique identities across multiple video frames.

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Dataset | COCO8 |
| Epochs | 10 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Device in recorded run | CPU |
| Confidence threshold | 0.25 |
| Validation set | 4 images, 17 instances |

## 6. Results

The supplied report records the following final validation metrics:

| Metric | Value |
|---|---:|
| Precision | 59.25% |
| Recall | 85.00% |
| mAP@0.50 | 87.18% |
| mAP@0.50:0.95 | 65.48% |
| Inference time | 317.5 ms/image |

Training losses decreased over the 10 epochs:

| Loss | Epoch 1 | Epoch 10 |
|---|---:|---:|
| Box loss | 0.9306 | 0.6447 |
| Classification loss | 2.5560 | 1.0071 |
| DFL loss | 1.3921 | 1.1146 |

For the sample CCTV-style image, the notebook reports **5 cars and 1 person** (6 detections total).

**Interpretation:** The experiment demonstrates the model-training, evaluation, detection and counting workflow. Since validation contains only four images and 17 instances from COCO8, these scores do not establish reliable performance on real campus CCTV footage.

## 7. Repository Contents

```text
project/
├── YOLO_deep_learning_30_09.ipynb
├── YOLO_Deep_Learning_Project_Report.pdf
├── YOLO_Deep_Learning_UML 1.pdf
└── README.md
```

The notebook generates training outputs such as `results.csv`, loss plots, model weights (`best.pt`, `last.pt`) and prediction images when executed.

## 8. Installation and Execution

Install the required packages:

```bash
pip install ultralytics opencv-python matplotlib pandas pyyaml jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `YOLO_deep_learning_30_09.ipynb` and run the cells in order. Alternatively, upload the notebook to [Google Colab](https://colab.research.google.com/) and execute it there.

**Note:** The notebook contains Google Colab paths (for example, `/content/...`). Update input and output paths when running locally. The recorded experiment used CPU execution; a compatible CUDA-enabled GPU may be configured if available.

## 9. Limitations and Future Work
- COCO8 is a small sample dataset, not campus-specific CCTV data.
- The validation set is too small to establish real-world reliability.
- The supplied notebook demonstrates image inference; a complete video pipeline needs to be implemented and tested.
- Current counting is per image/frame and does not provide unique-person tracking.

**Future improvements:** collect and annotate campus CCTV footage, train and test on separate campus-specific splits, implement video processing, add ByteTrack or SORT for object tracking, and build a dashboard for live counts and analytics.

## 10. References
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLO11 Model Documentation](https://docs.ultralytics.com/models/yolo11/)
- [COCO Dataset Documentation](https://docs.ultralytics.com/datasets/detect/coco/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

---
**Domain:** Deep Learning | Computer Vision | Object Detection

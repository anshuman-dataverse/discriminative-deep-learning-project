# Discriminative Deep Learning Project

**IE 7615 – Discriminative Deep Learning · Northeastern University · Fall 2026 · Group 1**

A deep learning system that (1) identifies the object in a single-object image by its Object ID, and (2) detects, identifies and locates every object in a multi-object image. The dataset is photographed by the class: each student contributes 100 images (224×224 RGB) of one unique object.

## Team

- Anshuman Singh (002590892)
- Rohan Prakash Krishna Prakash (002317798)
- Sree Ramya Pasala (002301508)

## Milestones

| Milestone | Task | Status | Report |
|---|---|---|---|
| [Milestone 1](milestone1/) | Single-object identification with CNNs | Done | [PDF](milestone1/report/Milestone1_Report.pdf) |
| [Milestone 2](milestone2/) | Multi-object detection and localization with YOLOv8 | Done | [PDF](milestone2/report/Milestone2_Report.pdf) |

---

## Milestone 1: Single-object identification

Four CNNs were trained and compared on the same stratified 70/15/15 split: 38 classes, 3,619 images, 535 test images.

| Model | Type | Val acc. | Test acc. | Macro F1 | Params | ms / image | Train time |
|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | Transfer learning | 99.81% | **99.25%** | **0.993** | 4.06 M | 10.9 | 12.0 min |
| MobileNetV3-Large | Transfer learning | 99.44% | 99.25% | 0.992 | 4.25 M | 7.1 | 7.7 min |
| ResNet18 | Transfer learning | 99.44% | 98.69% | 0.987 | 11.2 M | 4.7 | 8.4 min |
| SimpleCNN | From scratch (baseline) | 70.65% | 67.85% | 0.664 | 1.18 M | 4.3 | 25.7 min |

**Best model: EfficientNet-B0.** It misclassifies 4 of 535 test images, and all four are cluttered photos that also contain another class's object. Transfer learning is decisive: the scratch CNN reaches only 67.85% with about 67 training images per class.

Pipeline:

- **Data preparation:** audit the raw Drive uploads, fix sizes, names and EXIF orientation, remove duplicate photos, then make a stratified split.
- **Augmentation:** random resized crop, horizontal flip, ±15° rotation and color jitter.
- **Training:** AdamW with a cosine learning-rate schedule, label smoothing 0.1 and early stopping.
- **Evaluation:** one scoring of each model's best-validation checkpoint on the held-out test set.

### Structure

```
milestone1/
├── data/
│   ├── train/OBJ###/            2,549 images
│   ├── val/OBJ###/                535 images
│   └── test/OBJ###/               535 images
├── models/                      EfficientNetB0.pt, MobileNetV3.pt, ResNet18.pt, SimpleCNN.pt
├── notebooks/                   01_data_preparation, 02_train_models, 03_evaluate_and_demo
├── report/                      Milestone1_Report.pdf, Milestone1_Report.md
├── results/                     <Model>_classification_report.json, <Model>_confusion_matrix.png,
│                                <Model>_training_history.png, class_labels.json, model_comparison.{json,png}
├── screenshots/                 figures used in the report
└── scripts/
    ├── 01_data_inspection.py    audit the raw Drive download
    ├── 02_split_data.py         clean, deduplicate and split into data/{train,val,test}
    ├── 03_train_models.py       train the four models
    ├── 04_evaluate_models.py    compare models, pick the best
    ├── 05_predict.py            identify the object in new photos
    └── classifier/              shared code: data, models, train, predict, compare, plots
```

### Setup

```bash
git clone https://github.com/anshuman-dataverse/discriminative-deep-learning-project.git
cd discriminative-deep-learning-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

The split dataset is already in `milestone1/data/`, so training and prediction work straight after cloning:

```bash
python milestone1/scripts/03_train_models.py         # -> models/, results/
python milestone1/scripts/04_evaluate_models.py      # -> results/model_comparison.{json,png}
python milestone1/scripts/05_predict.py photo.jpg    # EfficientNetB0 by default
```

To rebuild the dataset from scratch, put the extracted shared-Drive download(s) in `milestone1/data/raw/` and run:

```bash
python milestone1/scripts/01_data_inspection.py
python milestone1/scripts/02_split_data.py
```

The notebooks in `milestone1/notebooks/` run the same pipeline step by step, with error analysis and a live demo. Notebook 02 skips models that are already trained; set `RETRAIN = True` to retrain. Training runs on CUDA, Apple MPS or CPU, and a full retrain takes about 55 minutes on an Apple M4.

---

## Milestone 2: Multi-object detection and localization

A COCO-pretrained YOLOv8s was fine-tuned on 1,700 generated multi-object images covering all **73 objects** (final Object IDs). Given an image, it returns the Object ID, confidence and bounding box of every object.

| Metric (250 test images, 941 objects) | Value |
|---|---|
| mAP@0.5 | **0.981** |
| mAP@0.5:0.95 | 0.978 |
| Precision / Recall | 0.969 / 0.930 |
| Objects with correct ID and location (conf ≥ 0.5) | 94.4% (0 false alarms) |
| Images with every object correct | 80.4% |
| Inference | 3.6 ms / image |

How the multi-object images are built:

- **Sources:** the full shared Drive (73 classes) is cleaned and split 70/15/15 per class with the Milestone 1 pipeline. Background-only photos are split the same way.
- **Composition:** 2–6 randomly chosen objects (no class twice) are pasted onto a background photo from the same split: scattered at random sizes (50%), concatenated into a 2×2 / 3×3 grid (25%), or packed into a collage where photos touch or overlap slightly (25%), with flip and colour augmentation.
- **Labels:** YOLO boxes are written automatically from the paste positions; `manifest.csv` records every source photo.
- **Checks:** each split uses only its own source photos; a path and image-fingerprint check confirms that no photo appears in more than one split, and every label is validated.

### Structure

```
milestone2/
├── data/
│   ├── singles/{train,val,test}/OBJ###/   4,886 / 1,028 / 1,028 single-object images
│   ├── singles/backgrounds/               373 background-only photos
│   └── multi/                             images/, labels/ (1,200 / 250 / 250), data.yaml, manifest.csv
├── models/                                yolov8s_best.pt
├── report/                                Milestone2_Report.pdf, Milestone2_Report.md
├── results/                               test_metrics.json, per_class_ap.csv, dataset_stats.json, curves
├── runs/yolov8s/                          Ultralytics training logs and plots
├── screenshots/                           figures used in the report
└── scripts/
    ├── 00_prepare_singles.py              clean and split the full Drive download
    ├── 01_make_multi_object.py            generate multi-object images + YOLO labels
    ├── 02_train_yolo.py                   fine-tune YOLOv8s
    ├── 03_evaluate_yolo.py                test metrics, example detections
    ├── 04_detect.py                       detect objects in new images
    ├── 05_export_web.py                   export models (ONNX) and results for the website
    ├── 06_report_figures.py               render console outputs as report figures
    └── detector/                          shared code: compose, evaluate, plots
```

### Usage

```bash
cd milestone2
python scripts/04_detect.py data/multi/images/test/test_0003.jpg   # IDs + boxes, annotated copy in runs/detect/
```

To rebuild everything, put the extracted shared-Drive download in `milestone2/data/raw/` and run `00` to `03` in order. Training takes about 2 hours on an Apple M4.

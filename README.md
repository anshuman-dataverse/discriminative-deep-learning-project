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
| [Milestone 3](website/) | Live demo: identify a single object; detect, identify and locate all objects | Live | [ie7615-group1-object-detection.vercel.app](https://ie7615-group1-object-detection.vercel.app) |

---

## Milestone 1: Single-object identification

Four CNNs were trained and compared on the same stratified 70/15/15 split of the full class dataset: **73 classes (final Object IDs)**, 6,942 images, 1,028 test images.

| Model | Type | Val acc. | Test acc. | Top-5 | Macro F1 | Params | ms / image | Train time |
|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | Transfer learning | 98.74% | **98.35%** | 100.00% | **0.983** | 4.10 M | 10.2 | 14.6 min |
| MobileNetV3-Large | Transfer learning | 98.25% | 97.96% | 99.90% | 0.980 | 4.30 M | 8.3 | 9.8 min |
| ResNet18 | Transfer learning | 98.35% | 97.86% | 99.81% | 0.979 | 11.21 M | 4.4 | 12.5 min |
| SimpleCNN | From scratch (baseline) | 63.52% | 64.98% | 86.77% | 0.624 | 1.19 M | 3.7 | 48.8 min |

**Best model: EfficientNet-B0.** It misclassifies 17 of 1,028 test images; 13 of those 17 are low-confidence guesses (below 0.6), and the hardest classes (OBJ013, OBJ061 vs OBJ059, OBJ041 vs OBJ048) are photos where the object is small in a wide scene. Transfer learning is decisive: the scratch CNN reaches only 64.98% with about 67 training images per class.

The submitted Milestone 1 report (October 4) covers the 38 classes uploaded at that time (EfficientNet-B0 99.25% on 535 test images). On October 7 all four models were retrained on the full 73-class download with the final Object IDs, which also corrected two IDs from the first upload (OBJ124 is final OBJ011, OBJ002 is final OBJ021); the table above and the demo website use the retrained models.

Pipeline:

- **Data preparation:** audit the raw Drive uploads, fix sizes, names and EXIF orientation, remove duplicate photos, then make a stratified split.
- **Augmentation:** random resized crop, horizontal flip, ±15° rotation and color jitter.
- **Training:** AdamW with a cosine learning-rate schedule, label smoothing 0.1 and early stopping.
- **Evaluation:** one scoring of each model's best-validation checkpoint on the held-out test set.

### Structure

```
milestone1/
├── data/
│   ├── train/OBJ###/            4,886 images
│   ├── val/OBJ###/              1,028 images
│   └── test/OBJ###/             1,028 images
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

To rebuild the dataset from scratch, put the extracted shared-Drive download(s) in `milestone1/data/raw/` (or pass their folder to `02_split_data.py`) and run:

```bash
python milestone1/scripts/01_data_inspection.py
python milestone1/scripts/02_split_data.py
```

The notebooks in `milestone1/notebooks/` run the same pipeline step by step, with error analysis and a live demo. Notebook 02 skips models that are already trained; set `RETRAIN = True` to retrain. Training runs on CUDA, Apple MPS or CPU, and a full retrain on the 73 classes takes about 1.5 hours on an Apple M4.

---

## Milestone 2: Multi-object detection and localization

A COCO-pretrained YOLOv8s was fine-tuned on 1,400 multi-object grid images covering **72 objects** (final Object IDs; OBJ054 excluded). Given an image, it returns the Object ID, confidence and bounding box of every object.

| Metric (200 test images, 2,960 objects) | Value |
|---|---|
| mAP@0.5 | **0.974** |
| mAP@0.5:0.95 | 0.925 |
| Precision / Recall | 0.956 / 0.951 |
| Objects with correct ID and location (conf ≥ 0.5) | 95.0% |
| Images with every object correct (4–25 objects each) | 32.0% |
| Inference | 33.8 ms / image (Tesla T4, imgsz 1120) |

How the multi-object images are built:

- **Sources:** the full shared Drive is cleaned with the Milestone 1 pipeline, OBJ054 is excluded, byte-identical duplicates are removed, and only then are the photos split 70/15/15 per class.
- **Object boxes:** Grounding DINO and OWLv2 each box the object in every single-object photo from a short text prompt; a photo is kept only when the two boxes agree (IoU ≥ 0.5), and the label is their mean (82.8% of photos kept, median IoU 0.963).
- **Grids:** 224×224 photos of distinct, randomly chosen objects are concatenated into 2×2, 3×3, 4×4, 5×4 or 5×5 grids (448×448 up to 1120×1120), with flip and colour augmentation; 1,000 / 200 / 200 images, 20,720 objects.
- **Checks:** each split uses only its own photos; a path and image-fingerprint check confirms that no photo appears in more than one split, and every label is validated.

### Structure

```
milestone2/
├── notebooks/milestone2_colab.ipynb       whole pipeline on Google Colab (checkpoints on Google Drive)
├── models/                                yolov8s_best.pt
├── report/                                Milestone2_Report.pdf, Milestone2_Report.md
├── results/                               test_metrics.json, per_class_ap.csv, dataset_stats.json, label_stats.json, curves
├── runs/                                  console logs, Ultralytics training and evaluation plots
├── screenshots/                           figures used in the report
└── scripts/
    ├── 00_prepare_singles.py              clean, deduplicate and split the Drive photos
    ├── 01_label_boxes.py                  object boxes from Grounding DINO + OWLv2, cross-checked
    ├── 02_make_grids.py                   grid images + YOLO labels, split and label checks
    ├── 03_train_yolo.py                   fine-tune YOLOv8s (imgsz 1120, batch 8)
    ├── 04_evaluate_yolo.py                test metrics, example detections, failure cases
    ├── 05_detect.py                       detect objects in new images
    ├── 06_export_web.py                   export models (ONNX) and results for the website
    ├── 07_report_figures.py               render console outputs as report figures
    └── detector/                          shared code: boxes, compose, evaluate, plots, prompts
```

### Usage

```bash
cd milestone2
python scripts/05_detect.py data/multi/images/test/test_0003.jpg   # IDs + boxes, annotated copy in runs/detect/
```

The image data is not in the repository. The Colab notebook downloads the shared photos and runs `00` to `05` in order; training takes about 52 minutes on a Tesla T4.

---

## Milestone 3: Live demo website

**https://ie7615-group1-object-detection.vercel.app**

Both trained models run in the visitor's browser with ONNX Runtime Web (WebGPU on the GPU, WebAssembly on the CPU otherwise), so there is no server and uploaded photos never leave the device.

- **Detect objects:** upload, paste or photograph a multi-object image, or pick a held-out test image; every object gets a box with its Object ID and confidence, and test images are scored correct or wrong.
- **Identify object:** the EfficientNet-B0 classifier gives the Object ID of a single-object photo with its top-5.
- **Build a grid:** choose a grid size and objects; the browser builds a grid like the training data and runs the detector.
- **Metrics, Method, Objects:** results of both milestones, the pipeline, and all 73 objects (the detector covers 72; OBJ054 is excluded from Milestone 2).

`milestone2/scripts/06_export_web.py` exports both models to ONNX (checked against PyTorch on test images) and the results into `website/`. To run the site locally:

```bash
cd website
npm install
npm run dev        # http://localhost:3000
```

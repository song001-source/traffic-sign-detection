# YOLO12 Traffic Sign Detection

[中文](README.md) | English

A traffic sign detection system based on [Ultralytics YOLO12](https://docs.ultralytics.com/models/yolo12/), trained on a **COCO-format dataset of 46 classes of Chinese traffic signs**. It supports three inference modes — **image, video, and live camera** — and ships with a complete web demo (image / video / browser-camera real-time detection).

> ⚠️ **Naming note**: Ultralytics officially names this model **`yolo12`** (not `yolov12`). The model file is `yolo12n.pt`. Many tutorials that write "YOLOv12" are inaccurate.

---

## 1. Hardware Requirements

This project is verified in the following environment:

| Component | Configuration |
|-----------|---------------|
| GPU | NVIDIA GPU (Blackwell architecture, 8GB+ VRAM, CUDA 12.8+) |
| PyTorch | 2.x+cu128 (**RTX 50-series requires the CUDA 12.8 build**) |
| Python | 3.11 |
| Node.js | 18+ (optional, for rebuilding the frontend) |

**Key point**: RTX 50-series GPUs use the Blackwell architecture (compute capability 12.0) and **require the cu128 build of PyTorch**, otherwise you will get a `no kernel image is available` error. For other GPUs, install the PyTorch build matching your CUDA version.

---

## 2. Installation

### 2.1 Create a conda environment

```bash
conda create -n yolov12 python=3.11 -y
conda activate yolov12
```

### 2.2 Install PyTorch (CUDA 12.8 build — required for RTX 50-series)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2.3 Install other dependencies

```bash
pip install -r requirements.txt
```

### 2.4 Verify the environment

```bash
python src/verify_env.py
```

All 5 checks should pass. In particular, confirm that `[3/5] GPU operator availability` passes.

---

## 3. Project Structure

```
traffic-sign-detection/
├── inference/               # Inference core — pure YOLO, zero web dependencies
│   ├── engine.py            # DetectorService (model loading / image / video / frame inference)
│   ├── config.py            # InferenceConfig (thresholds / image size / device)
│   ├── schemas.py           # Detection Pydantic dataclasses
│   ├── label_mapping.py     # Chinese name mapping get_chinese_name()
│   └── coco_names.json      # 46-class bilingual class name table
├── demo/                    # Demo system — FastAPI, depends on inference one-way
│   ├── main.py              # FastAPI entry (auto-mounts frontend/dist/)
│   ├── config.py            # ServiceConfig (host / port / paths / Redis)
│   ├── schemas.py           # API response models
│   ├── routers/
│   │   ├── detect.py        # Image/video detection, progress polling, result download
│   │   ├── models.py        # Model list / switching / class names
│   │   └── stream.py        # WebSocket real-time stream inference
│   └── tasks/
│       └── video_tasks.py   # Celery async video tasks (optional, requires Redis)
├── frontend/                # Vue 3 + Vite + Element Plus + ECharts
│   └── src/views/           # ImageDetect / VideoDetect / CameraDetect
├── src/                     # CLI scripts (independent of the web system)
│   ├── verify_env.py        # Environment verification script
│   ├── train.py             # Training script (default: configs/coco.yaml)
│   ├── predict_image.py     # Image inference
│   ├── predict_video.py     # Video inference
│   └── predict_camera.py    # Camera real-time detection
├── configs/
│   └── coco.yaml            # Dataset config (46 classes, relative path)
├── data/
│   └── coco/                # Dataset (needs to be prepared by yourself, see below)
├── models/                  # Custom weights
├── run.py                   # One-command startup for API + frontend
├── best.pt                  # Trained weights (default inference model)
├── yolo12n.pt               # Official pretrained weights (fine-tuning / retraining)
├── requirements.txt
└── README.md
```

---

## 4. Usage

### 4.1 Inference (works out of the box)

The repository includes the trained `best.pt`. After cloning and setting up the environment, you can infer directly:

```bash
# Image inference
python src/predict_image.py --source your_image.jpg

# Video inference
python src/predict_video.py --source your_video.mp4

# Camera real-time detection (press q to quit, s to save a snapshot)
python src/predict_camera.py
```

Results are saved to `runs/` by default. The web demo automatically prefers `best.pt`.

### 4.2 Training

```bash
# Smoke test (3 epochs, quick pipeline check)
python src/train.py --smoke

# Full training (80 epochs)
python src/train.py

# Custom parameters
python src/train.py --model yolo12s.pt --epochs 50 --batch 8
```

After training, the best weights are at `runs/train-*/weights/best.pt` (the inference service auto-discovers and loads the newest best.pt).

---

## 5. Dataset Preparation

> The dataset is **not included in this repository** (too large). You need to prepare it yourself. The expected layout is:

```
data/coco/
├── images/
│   ├── train/    # Training images (jpg)
│   └── val/      # Validation images (jpg)
└── labels/
    ├── train/    # Training labels (YOLO txt, same basename as images)
    └── val/      # Validation labels (YOLO txt, same basename as images)
```

Labels use YOLO format: one line per object — `class_id x_center y_center width height` (normalized coordinates), e.g.:

```
45 0.534722 0.453704 0.182292 0.177469
```

Class IDs follow the 46-class order in `configs/coco.yaml` (ID 0-45); see the full table at the end of this file.

**Data source reference**: this project is organized and extended from the [TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/) (Tsinghua-Tencent) public traffic sign dataset — please respect its license. You can also source data from platforms like [Roboflow](https://universe.roboflow.com/) or annotate your own.

**Config note**: `configs/coco.yaml` uses the relative path `../data/coco`, so after cloning and placing the data you can train directly without editing any config.

---

## 6. Web Demo

The project includes a complete web demo supporting image, video, and real-time camera detection.

### 6.1 Quick start

```bash
# One-command start (API + frontend)
python run.py

# Use a different port if occupied
python run.py --port 18000

# Force-rebuild the frontend
python run.py --build

# API-only mode (skip frontend check)
python run.py --no-frontend
```

Then open `http://localhost:8000/` in your browser.

### 6.2 Features

| Page | Description |
|------|-------------|
| **Image detection** | Drag-and-drop upload, shows original vs. annotated comparison, list of detected signs with confidence |
| **Video detection** | Upload a video, processed asynchronously in a background thread (`/video/process` + progress polling, no Redis needed), produces an annotated video |
| **Live camera** | Browser camera streamed over WebSocket for real-time detection with live boxes |

### 6.3 API docs

After starting the service, visit `http://localhost:8000/api/docs` for interactive API documentation with online testing.

### 6.4 Tech stack

| Layer | Technology |
|-------|------------|
| Frontend | Vue 3 + Vite + Element Plus + ECharts |
| Backend | FastAPI + Uvicorn |
| Inference | Ultralytics YOLO12 (`inference/` core; `demo/` depends on it one-way) |
| Video tasks | Thread-based progress endpoint used by default (no Redis); Celery version (`/video/async`) optional, requires Redis |
| WebSocket | Real-time stream inference |

---

## 7. FAQ

### Q1: `no kernel image is available for execution`
**Cause**: PyTorch does not support your GPU architecture (Blackwell).
**Fix**: reinstall the cu128 build of PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### Q2: Downloading `yolov12n.pt` returns 404
**Cause**: wrong filename. The correct name is `yolo12n.pt` (no "v").
**Fix**: download from `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n.pt`

### Q3: CUDA out of memory during training
**Cause**: insufficient VRAM (8GB).
**Fix**: lower the batch size: `python src/train.py --batch 8` or `--batch 4`

### Q4: flash-attn installation fails
**Note**: YOLO12 uses flash-attn acceleration on Linux, but there is no prebuilt version for Windows. **Do not install flash-attn** — ultralytics falls back to standard attention on Windows automatically; functionality is unaffected.

### Q5: Camera cannot be opened
**Fix**: an external USB camera may be device index 1; try `python src/predict_camera.py --cam 1`

### Q6: `Errno 10048` when starting the server (port in use)
**Cause**: port 8000 is occupied by another program (e.g. **Steam's steamwebhelper.exe** often uses the localhost 8000-8009 range).
**Fix**: start on another port, e.g.:
```bash
python run.py --port 18000
```

---

## 8. 46 Traffic Sign Classes

The authoritative class definitions are in `configs/coco.yaml` and `inference/coco_names.json` (both consistent):

| ID | Code | Meaning | ID | Code | Meaning |
|----|------|---------|----|------|---------|
| 0 | pl80 | Speed limit (80 km/h) | 23 | pn | No parking |
| 1 | p6 | No entry for non-motorized vehicles | 24 | w55 | Watch for children |
| 2 | p5 | No U-turn | 25 | p26 | No entry for trucks |
| 3 | pm55 | Weight limit (55 t) | 26 | p13 | Street name sign |
| 4 | pl60 | Speed limit (60 km/h) | 27 | pr40 | Speed limit ends (40 km/h) |
| 5 | ip | Pedestrian crossing | 28 | pl20 | Speed limit (20 km/h) |
| 6 | p11 | No honking | 29 | pm30 | Weight limit (30 t) |
| 7 | i2r | Non-motorized vehicles only | 30 | pl40 | Speed limit (40 km/h) |
| 8 | p23 | No left turn | 31 | i2 | Motor vehicles only |
| 9 | pg | Yield | 32 | pl120 | Speed limit (120 km/h) |
| 10 | il80 | Minimum speed (80 km/h) | 33 | w32 | Unguarded railway crossing |
| 11 | ph4 | Height limit (4 m) | 34 | ph5 | Height limit (5 m) |
| 12 | i4 | Keep right | 35 | il60 | Minimum speed (60 km/h) |
| 13 | pl70 | Speed limit (70 km/h) | 36 | w57 | Watch for pedestrians |
| 14 | pne | No entry | 37 | pl100 | Speed limit (100 km/h) |
| 15 | ph4.5 | Height limit (4.5 m) | 38 | w59 | Merging traffic |
| 16 | p12 | No entry for motorcycles | 39 | il100 | Minimum speed (100 km/h) |
| 17 | p3 | No entry for large buses | 40 | p19 | No right turn |
| 18 | pl5 | Speed limit (5 km/h) | 41 | pm20 | Weight limit (20 t) |
| 19 | w13 | Dangerous mountain road | 42 | i5 | Keep left |
| 20 | i4l | Keep left | 43 | p27 | No entry for dangerous goods vehicles |
| 21 | pl30 | Speed limit (30 km/h) | 44 | pl50 | Speed limit (50 km/h) |
| 22 | p10 | No entry for motor vehicles | 45 | wo | Other |

---

## 9. License

This project is open-sourced under the MIT License — see [LICENSE](LICENSE).

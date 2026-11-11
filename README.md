# SignSense AI - Real-Time Sign Language Detection & Translator

A deep learning web application for real-time American Sign Language (ASL) detection and translation powered by **YOLOv5s** and **PyTorch**, featuring an interactive modern web UI.

---

## 🌟 Key Features

1. **🎥 Real-Time Webcam Detection**:
   - **In-Browser WebCam Stream**: Smooth client-side canvas capture streamed to the backend (~20+ FPS on CPU, even faster on GPU).
   - **Desktop OpenCV Camera Stream**: Direct hardware camera stream with MJPEG support.
   - **Real-Time HUD**: Live FPS counter, inference latency in milliseconds, active detection count, and confidence bar.
   - **Smart Sentence Builder**: Holds gestures for 0.5s to automatically build words and sentences.
   - **🔊 Text-to-Speech (TTS)**: Translates signed sentences into spoken audio using speech synthesis.
   - **Adjustable Controls**: Real-time sliders for Confidence Threshold (10% - 95%) and IoU / NMS threshold.

2. **📸 Image Upload & Analysis**:
   - **Drag & Drop / File Picker**: Upload photos in JPG, PNG, or WEBP format.
   - **One-Click Quick Test Gallery**: Built-in interactive gallery with sample test images for all 36 classes (letters A-Z and digits 0-9).
   - **Comparison Toggle**: Seamlessly switch between the annotated detection view and original image.
   - **Detailed Metrics Table**: Displays predicted sign, confidence score, bounding box pixel coordinates `[X1, Y1, X2, Y2]`, and box dimensions.
   - **Export**: One-click download of the annotated detection image.

3. **📊 Model Analytics & Sign Dictionary**:
   - **Full 36-Class Visual Dictionary**: Interactive visual guide of all 36 gestures with instant search and click-to-test capability.
   - **Training Performance Curves**: High-resolution charts from the 300-epoch Colab training run:
     - Validation & Test Confusion Matrices
     - Precision-Recall (PR) Curves
     - F1-Score Confidence Curves
     - Precision & Recall curves
   - **Click-to-Zoom**: Interactive modal to inspect training graphs in full resolution.

---

## 🚀 Quick Start Guide

### Option 1: Double-Click Launcher (Windows)
Double-click `run.bat` in the project root folder.

### Option 2: Command Line
```bash
python start_app.py
```

Then open your web browser at:
```
http://localhost:5000
```

---

## 📋 Supported Classes (36 Total)

| Category | Classes |
| :--- | :--- |
| **Alphabet (A-Z)** | `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z` |
| **Digits (0-9)** | `Zero`, `One`, `Two`, `Three`, `Four`, `Five`, `Six`, `Seven`, `Eight`, `Nine` |

---

## 🧠 Model Architecture & Specifications

- **Model Backbone**: YOLOv5s (PyTorch)
- **Input Resolution**: 416 × 416 px
- **Parameters**: 7,107,217 (~7.1 Million)
- **FLOPs**: 16.1 GFLOPs
- **Trained Epochs**: 300 epochs on Google Colab GPU
- **Inference Speed**: ~45-50ms per frame on CPU (20+ FPS)

---

## 📂 Project Structure

```
SignLanguageDetection3/
│
├── app.py                     # Flask web server & REST API endpoints
├── detector.py                # YOLOv5 inference engine & cross-platform loader
├── camera.py                  # OpenCV direct camera capture & streaming
├── start_app.py               # Launcher script with auto-browser opening
├── run.bat                    # 1-click Windows batch launcher
├── test_server.py             # Automated end-to-end verification script
│
├── SignLanguageDetection3/    # Original training assets & weights
│   ├── model/best.pt          # PyTorch trained weights
│   ├── data/data.yaml         # Roboflow dataset configuration
│   ├── results/               # Training charts & metrics
│   └── notebooks/             # Colab training notebook
│
├── yolov5/                    # YOLOv5 repository
├── static/
│   ├── css/style.css          # Dark-mode glassmorphic styling
│   ├── js/app.js              # Real-time streaming, HUD & TTS logic
│   ├── samples/               # 36 distinct sample images for instant testing
│   └── results/               # Evaluation charts (Confusion matrix, PR, F1)
│
└── templates/
    └── index.html             # Single-page web dashboard
```

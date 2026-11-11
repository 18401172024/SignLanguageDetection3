"""
Sign Language Detection Model Wrapper
Supports YOLOv5 model trained on American Sign Language (36 classes: A-Z, 0-9).
"""

import sys
import os
import pathlib
import time
import base64
import cv2
import numpy as np
import torch

# Direct Ultralytics and Matplotlib to local workspace cache directories
_workspace_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["YOLO_CONFIG_DIR"] = os.path.join(_workspace_dir, ".ultralytics")
os.environ["MPLCONFIGDIR"] = os.path.join(_workspace_dir, ".mplconfig")
os.environ["YOLO_VERBOSE"] = "False"
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Fix Windows/Linux PosixPath cross-deserialization issue
pathlib.PosixPath = pathlib.WindowsPath

# Safely handle any restricted permission checks during imports
_orig_exists = pathlib.Path.exists
def _safe_exists(self):
    try:
        return _orig_exists(self)
    except (PermissionError, OSError):
        return False
pathlib.Path.exists = _safe_exists

_orig_is_dir = pathlib.Path.is_dir
def _safe_is_dir(self):
    try:
        return _orig_is_dir(self)
    except (PermissionError, OSError):
        return False
pathlib.Path.is_dir = _safe_is_dir

_orig_is_file = pathlib.Path.is_file
def _safe_is_file(self):
    try:
        return _orig_is_file(self)
    except (PermissionError, OSError):
        return False
pathlib.Path.is_file = _safe_is_file

class PathLibLocal:
    PosixPath = pathlib.WindowsPath
    WindowsPath = pathlib.WindowsPath
    Path = pathlib.WindowsPath

sys.modules['pathlib._local'] = PathLibLocal

# Distinct neon/pastel color palette for 36 classes (BGR for OpenCV)
CLASS_COLORS = [
    (255, 99, 132),   # A
    (54, 162, 235),   # B
    (255, 206, 86),   # C
    (75, 192, 192),   # D
    (153, 102, 255),  # E
    (255, 159, 64),   # Eight
    (46, 204, 113),   # F
    (231, 76, 60),    # Five
    (52, 152, 219),   # Four
    (155, 89, 182),   # G
    (241, 196, 15),   # H
    (230, 126, 34),   # I
    (26, 188, 156),   # J
    (236, 240, 241),  # K
    (149, 165, 166),  # L
    (243, 156, 18),   # M
    (189, 195, 199),  # N
    (39, 174, 96),    # Nine
    (41, 128, 185),   # O
    (142, 68, 173),   # One
    (192, 57, 43),    # P
    (211, 84, 0),     # Q
    (22, 160, 133),   # R
    (127, 140, 141),  # S
    (245, 112, 190),  # Seven
    (123, 237, 159),  # Six
    (255, 71, 87),    # T
    (112, 161, 255),  # Three
    (255, 165, 2),    # Two
    (47, 53, 66),     # U
    (164, 176, 190),  # V
    (116, 125, 140),  # W
    (255, 107, 129),  # X
    (106, 176, 76),   # Y
    (34, 166, 179),   # Z
    (255, 121, 121),  # Zero
]


class SignLanguageDetector:
    def __init__(self, model_path=None, yolov5_dir=None, device="cpu"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        if model_path is None:
            # Look in standard locations
            candidate_paths = [
                os.path.join(base_dir, "SignLanguageDetection3", "model", "best.pt"),
                os.path.join(base_dir, "model", "best.pt"),
                os.path.join(base_dir, "..", "SignLanguageDetection3", "model", "best.pt"),
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    model_path = p
                    break
        
        if yolov5_dir is None:
            candidate_yolo = [
                os.path.join(base_dir, "yolov5"),
                os.path.join(base_dir, "..", "yolov5"),
            ]
            for p in candidate_yolo:
                if os.path.exists(p):
                    yolov5_dir = p
                    break

        self.model_path = model_path
        self.yolov5_dir = yolov5_dir
        self.device = device
        self.model = None
        self.class_names = []
        self._load_model()

    def _load_model(self):
        print(f"Loading Sign Language Detection Model from: {self.model_path}")
        print(f"Using YOLOv5 repository at: {self.yolov5_dir}")
        
        try:
            self.model = torch.hub.load(
                self.yolov5_dir,
                "custom",
                path=self.model_path,
                source="local",
                device=self.device
            )
            # Default parameters
            self.model.conf = 0.35
            self.model.iou = 0.45
            self.class_names = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
            print(f"Model loaded successfully with {len(self.class_names)} classes.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def get_color(self, class_id):
        bgr = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        # Return both BGR tuple and Hex string
        hex_color = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
        return bgr, hex_color

    def detect(self, img_bgr, conf_thres=0.35, iou_thres=0.45, img_size=416):
        """
        Runs object detection on a BGR OpenCV image.
        Returns:
            annotated_img_bgr: OpenCV image with bounding boxes & labels drawn
            detections: List of detection dictionaries
            latency_ms: float (inference time in ms)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        self.model.conf = float(conf_thres)
        self.model.iou = float(iou_thres)

        t0 = time.time()
        # Convert BGR to RGB for YOLOv5 model
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=img_size)
        latency_ms = (time.time() - t0) * 1000.0

        detections = []
        df = results.pandas().xyxy[0]

        h, w = img_bgr.shape[:2]
        annotated_img = img_bgr.copy()

        for _, row in df.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = float(row['confidence'])
            class_id = int(row['class'])
            class_name = str(row['name'])

            # Clamp coordinates
            xmin = max(0, min(xmin, w - 1))
            ymin = max(0, min(ymin, h - 1))
            xmax = max(0, min(xmax, w - 1))
            ymax = max(0, min(ymax, h - 1))

            bgr_color, hex_color = self.get_color(class_id)

            detections.append({
                "class_id": class_id,
                "name": class_name,
                "confidence": round(conf, 4),
                "confidence_pct": round(conf * 100, 1),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": xmax - xmin,
                "height": ymax - ymin,
                "color": hex_color
            })

            # Draw visual bounding box with modern stylish look
            self._draw_box(annotated_img, xmin, ymin, xmax, ymax, class_name, conf, bgr_color)

        return annotated_img, detections, round(latency_ms, 2)

    def _draw_box(self, img, xmin, ymin, xmax, ymax, label, conf, color):
        """Draws bounding box with stylized rounded corner label banner."""
        thickness = max(2, int(min(img.shape[:2]) / 250))
        
        # Bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        # Label tag
        label_text = f"{label} {int(conf * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, thickness * 0.25)
        text_thickness = max(1, int(thickness * 0.75))

        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)
        
        # Determine label position
        tag_ymin = max(0, ymin - th - baseline - 10)
        tag_ymax = ymin
        tag_xmax = min(img.shape[1], xmin + tw + 16)

        # Draw filled label background
        cv2.rectangle(img, (xmin, tag_ymin), (tag_xmax, tag_ymax), color, -1)

        # High-contrast text (dark text on bright colors, white on dark)
        luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)

        text_x = xmin + 8
        text_y = tag_ymax - baseline - 4
        cv2.putText(img, label_text, (text_x, text_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    def detect_base64(self, base64_data, conf_thres=0.35, iou_thres=0.45, img_size=416):
        """Decode base64 image string, run detection, return annotated base64 and detections."""
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]

        img_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Could not decode image from base64 data.")

        annotated_bgr, detections, latency_ms = self.detect(
            img_bgr, conf_thres=conf_thres, iou_thres=iou_thres, img_size=img_size
        )

        # Encode back to JPEG base64
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return {
            "annotated_image": annotated_b64,
            "detections": detections,
            "count": len(detections),
            "latency_ms": latency_ms,
            "width": img_bgr.shape[1],
            "height": img_bgr.shape[0]
        }

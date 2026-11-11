"""
Sign Language Detection Web Application
Provides real-time browser webcam detection, server-side video streaming,
image upload analysis, sample test gallery, and model metrics viewer.
"""

import os
import io
import base64
import time
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from detector import SignLanguageDetector
from camera import VideoCamera

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Initialize Detector and Camera
print("Initializing Sign Language Detector...")
detector = SignLanguageDetector()
camera = VideoCamera(detector=detector)


@app.route("/")
def index():
    samples_dir = os.path.join(app.static_folder, "samples")
    class_thumbnails = {}
    if os.path.exists(samples_dir):
        for f in os.listdir(samples_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                cls_name = f.split("_")[0]
                if cls_name not in class_thumbnails:
                    class_thumbnails[cls_name] = f"/static/samples/{f}"
    return render_template("index.html", classes=detector.class_names, class_thumbnails=class_thumbnails)


@app.route("/api/model_info", methods=["GET"])
def model_info():
    """Return model architecture and class metadata."""
    chart_list = [
        {"title": "Validation Confusion Matrix", "url": "/static/results/validation/confusion_matrix.png", "desc": "Class-wise validation accuracy & confusion matrix"},
        {"title": "Validation F1-Confidence Curve", "url": "/static/results/validation/F1_curve.png", "desc": "F1 score across confidence thresholds"},
        {"title": "Validation Precision-Recall Curve", "url": "/static/results/validation/PR_curve.png", "desc": "Precision-Recall trade-off"},
        {"title": "Test Confusion Matrix", "url": "/static/results/test/confusion_matrix.png", "desc": "Final test set confusion matrix"},
        {"title": "Test PR Curve", "url": "/static/results/test/PR_curve.png", "desc": "Test set precision-recall curve"},
        {"title": "Test F1 Curve", "url": "/static/results/test/F1_curve.png", "desc": "Test set F1 curve"},
    ]

    return jsonify({
        "status": "success",
        "architecture": "YOLOv5s (PyTorch)",
        "input_resolution": "416x416",
        "classes_count": len(detector.class_names),
        "classes": detector.class_names,
        "parameters": "7.1M",
        "flops": "16.1 GFLOPs",
        "charts": chart_list
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return list of sample test images available."""
    samples_dir = os.path.join(app.static_folder, "samples")
    samples = []
    if os.path.exists(samples_dir):
        files = sorted(os.listdir(samples_dir))
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                cls_name = f.split("_")[0]
                samples.append({
                    "filename": f,
                    "class_name": cls_name,
                    "url": f"/static/samples/{f}"
                })
    return jsonify({"status": "success", "samples": samples})


@app.route("/api/detect_image", methods=["POST"])
def detect_image():
    """
    Detect sign language in an uploaded image.
    Accepts multipart file or JSON with base64 string.
    """
    try:
        conf_thres = float(request.form.get("confidence", request.json.get("confidence", 0.35) if request.is_json else 0.35))
        iou_thres = float(request.form.get("iou", request.json.get("iou", 0.45) if request.is_json else 0.45))
        img_bgr = None

        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"status": "error", "message": "No file selected"}), 400
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif request.is_json and "image" in request.json:
            b64_str = request.json["image"]
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif "sample_path" in request.form or (request.is_json and "sample_path" in request.json):
            rel_path = request.form.get("sample_path", request.json.get("sample_path", "") if request.is_json else "")
            full_path = os.path.join(app.root_path, rel_path.lstrip("/\\"))
            if os.path.exists(full_path):
                img_bgr = cv2.imread(full_path)

        if img_bgr is None:
            return jsonify({"status": "error", "message": "Invalid or missing image"}), 400

        # Encode original image to base64 for preview comparison
        _, orig_buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        orig_b64 = "data:image/jpeg;base64," + base64.b64encode(orig_buf).decode("utf-8")

        # Run inference
        annotated_bgr, detections, latency_ms = detector.detect(
            img_bgr, conf_thres=conf_thres, iou_thres=iou_thres, img_size=416
        )

        # Encode annotated image
        _, ann_buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ann_b64 = "data:image/jpeg;base64," + base64.b64encode(ann_buf).decode("utf-8")

        return jsonify({
            "status": "success",
            "original_image": orig_b64,
            "annotated_image": ann_b64,
            "detections": detections,
            "count": len(detections),
            "latency_ms": latency_ms,
            "dimensions": {"width": img_bgr.shape[1], "height": img_bgr.shape[0]}
        })

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/detect_video", methods=["POST"])
def detect_video():
    """
    Process an uploaded video file (MP4, WEBM, AVI, MOV) frame by frame,
    produce an annotated video output and a timestamped gesture timeline.
    """
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No video file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "Empty filename"}), 400

        conf_thres = float(request.form.get("confidence", 0.35))
        iou_thres = float(request.form.get("iou", 0.45))

        outputs_dir = os.path.join(app.static_folder, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        temp_input = os.path.join(outputs_dir, f"input_{timestamp}.mp4")
        temp_output = os.path.join(outputs_dir, f"annotated_{timestamp}.mp4")
        file.save(temp_input)

        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            if os.path.exists(temp_input): os.remove(temp_input)
            return jsonify({"status": "error", "message": "Could not open uploaded video"}), 400

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames = min(total_frames if total_frames > 0 else 300, 300)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        timeline = []
        frame_idx = 0
        t0 = time.time()

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            ann_frame, detections, _ = detector.detect(
                frame, conf_thres=conf_thres, iou_thres=iou_thres, img_size=416
            )
            writer.write(ann_frame)

            curr_sec = round(frame_idx / fps, 1)
            for det in detections:
                timeline.append({
                    "second": curr_sec,
                    "frame": frame_idx,
                    "sign": det["name"],
                    "confidence": det["confidence_pct"],
                    "color": det["color"]
                })

            frame_idx += 1

        cap.release()
        writer.release()
        if os.path.exists(temp_input):
            try: os.remove(temp_input)
            except: pass

        processing_time = round(time.time() - t0, 2)

        # Consolidate timeline events
        condensed_timeline = []
        for item in timeline:
            if not condensed_timeline or condensed_timeline[-1]["sign"] != item["sign"] or (item["second"] - condensed_timeline[-1]["second"]) > 0.8:
                condensed_timeline.append(item)

        return jsonify({
            "status": "success",
            "video_url": f"/static/outputs/{os.path.basename(temp_output)}",
            "timeline": condensed_timeline,
            "total_frames_processed": frame_idx,
            "fps": round(fps, 1),
            "processing_time_s": processing_time
        })

    except Exception as e:
        print(f"Video detection error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/detect_frame", methods=["POST"])
def detect_frame():
    """
    Fast endpoint for in-browser webcam streaming.
    Receives base64 video frame and returns annotated frame + detections.
    """
    try:
        data = request.get_json(silent=True)
        if not data or "frame" not in data:
            return jsonify({"status": "error", "message": "No frame data provided"}), 400

        conf_thres = float(data.get("confidence", 0.35))
        iou_thres = float(data.get("iou", 0.45))
        return_annotated = data.get("return_annotated", True)

        res = detector.detect_base64(
            data["frame"],
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            img_size=416
        )
        res["status"] = "success"
        return jsonify(res)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    """Start server-side desktop camera."""
    conf = float(request.json.get("confidence", 0.35)) if request.is_json else 0.35
    iou = float(request.json.get("iou", 0.45)) if request.is_json else 0.45
    camera.conf_thres = conf
    camera.iou_thres = iou
    success = camera.start()
    return jsonify({"status": "success" if success else "error", "running": camera.is_running})


@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    """Stop server-side desktop camera."""
    camera.stop()
    return jsonify({"status": "success", "running": False})


@app.route("/api/camera/status", methods=["GET"])
def camera_status():
    """Get server camera status, FPS, and current detections."""
    return jsonify(camera.get_status())


def gen_frames():
    """Generator for MJPEG server video feed."""
    while camera.is_running:
        frame_bytes = camera.get_jpeg_frame()
        if frame_bytes is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.01)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint for server webcam."""
    if not camera.is_running:
        camera.start()
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

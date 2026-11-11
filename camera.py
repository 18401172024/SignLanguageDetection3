"""
OpenCV Video Camera Streaming Module with threading and detection integration.
"""

import threading
import time
import cv2

class VideoCamera:
    def __init__(self, detector=None, camera_index=0):
        self.detector = detector
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_detections = []
        self.latest_latency = 0.0
        self.fps = 0.0
        self.conf_thres = 0.40
        self.iou_thres = 0.45
        self.thread = None

    def start(self):
        with self.lock:
            if self.is_running:
                return True
            try:
                # DirectShow backend on Windows is fast and stable
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    # Fallback to default backend
                    self.cap = cv2.VideoCapture(self.camera_index)
                
                if not self.cap.isOpened():
                    print(f"Warning: Could not open camera {self.camera_index}")
                    return False
                
                # Configure resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self.is_running = True
                self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.thread.start()
                print("Server webcam started successfully.")
                return True
            except Exception as e:
                print(f"Error starting camera: {e}")
                self.is_running = False
                return False

    def _capture_loop(self):
        prev_time = time.time()
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Run detection if detector is provided
            if self.detector is not None:
                try:
                    annotated_bgr, dets, lat_ms = self.detector.detect(
                        frame,
                        conf_thres=self.conf_thres,
                        iou_thres=self.iou_thres,
                        img_size=416
                    )
                    with self.lock:
                        self.latest_frame = annotated_bgr
                        self.latest_detections = dets
                        self.latest_latency = lat_ms
                except Exception as e:
                    with self.lock:
                        self.latest_frame = frame
            else:
                with self.lock:
                    self.latest_frame = frame

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            prev_time = curr_time
            time.sleep(0.01)

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.latest_frame = None
            print("Server webcam stopped.")

    def get_jpeg_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return jpeg.tobytes()
            return None

    def get_status(self):
        with self.lock:
            return {
                "is_running": self.is_running,
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latest_latency, 1),
                "detections": self.latest_detections,
                "count": len(self.latest_detections)
            }

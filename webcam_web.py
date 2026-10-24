"""Local browser interface for live Ultimate Frisbee player tracking."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_sort import nn_matching, preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from yolov10.config import YOLOv10Config
from yolov10.detection_adapter import DetectionAdapter
from yolov10.inference import YOLOv10Inference
from yolov10.model_loader import ModelLoader


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
WEIGHTS_PATH = PROJECT_ROOT / "checkpoints" / "yolov10s.pt"
ALLOWED_INPUT_SIZES = {416, 640, 960, 1280}
ALLOWED_PERSON_CONFIDENCES = {0.25, 0.4, 0.55}
MAX_FRAME_BYTES = 8 * 1024 * 1024
LOGGER = logging.getLogger("webcam_web")


class TrackingService:
    """Owns the persistent detector and DeepSORT tracker for one browser session."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._initializer: threading.Thread | None = None
        self._inference: YOLOv10Inference | None = None
        self._adapter: DetectionAdapter | None = None
        self._tracker: Tracker | None = None
        self._state: dict[str, Any] = {
            "status": "stopped",
            "error": None,
            "fps": 0.0,
            "players": 0,
            "frisbees": 0,
            "white_ovals": 0,
            "sports_balls": 0,
            "input_size": None,
            "device": None,
            "person_confidence": None,
        }

    def get_state(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._state)

    def start(
        self, input_size: int, device: str, person_confidence: float = 0.4
    ) -> None:
        if input_size not in ALLOWED_INPUT_SIZES:
            raise ValueError(
                "Inference size must be one of 416, 640, 960, or 1280."
            )
        if device not in {"cpu", "auto"}:
            raise ValueError("Device must be cpu or auto.")
        if person_confidence not in ALLOWED_PERSON_CONFIDENCES:
            raise ValueError(
                "Person confidence must be 0.25 (most detections), "
                "0.4 (balanced), or 0.55 (stricter)."
            )
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

        with self._state_lock:
            if self._state["status"] in {"loading", "running"}:
                raise RuntimeError("Tracking is already starting or running.")
            self._stop_event.clear()
            self._inference = None
            self._adapter = None
            self._tracker = None
            self._state = {
                "status": "loading",
                "error": None,
                "fps": 0.0,
                "players": 0,
                "frisbees": 0,
                "white_ovals": 0,
                "input_size": input_size,
                "device": device,
                "person_confidence": person_confidence,
            }

        self._initializer = threading.Thread(
            target=self._initialize_model,
            args=(input_size, device, person_confidence),
            name="webcam-model-initializer",
            daemon=True,
        )
        self._initializer.start()

    def _initialize_model(
        self, input_size: int, device: str, person_confidence: float
    ) -> None:
        try:
            config = YOLOv10Config(device=device, input_size=input_size)
            model_loader = ModelLoader(
                model_variant="yolov10s",
                device=config.device,
                confidence_threshold=config.confidence_threshold,
                iou_threshold=config.iou_threshold,
            )
            model = model_loader.load_model(str(WEIGHTS_PATH))
            model_loader.optimize_for_inference()
            inference = YOLOv10Inference(
                model, config, person_confidence=person_confidence
            )
            adapter = DetectionAdapter(
                target_classes=["person", "frisbee", "sports ball"],
                confidence_threshold=config.confidence_threshold,
            )
            tracker = Tracker(
                nn_matching.NearestNeighborDistanceMetric(
                    "cosine", 0.4, None
                ),
                max_age=30,
                n_init=2,
            )

            with self._state_lock:
                if self._stop_event.is_set():
                    self._state["status"] = "stopped"
                    return
                self._inference = inference
                self._adapter = adapter
                self._tracker = tracker
                self._state["status"] = "running"
        except Exception as error:
            LOGGER.exception("Could not initialize webcam detector.")
            with self._state_lock:
                self._state["status"] = "error"
                self._state["error"] = str(error)

    @staticmethod
    def _appearance_feature(frame: np.ndarray, tlwh: np.ndarray) -> np.ndarray:
        x, y, width, height = map(int, tlwh)
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(frame_width - 1, x))
        y1 = max(0, min(frame_height - 1, y))
        x2 = max(x1 + 1, min(frame_width, x + width))
        y2 = max(y1 + 1, min(frame_height, y + height))
        crop = frame[y1:y2, x1:x2]
        histogram = cv2.calcHist(
            [crop],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 256, 0, 256, 0, 256],
        )
        histogram = cv2.normalize(histogram, None).reshape(-1)
        return histogram.astype(np.float32)

    @staticmethod
    def _draw_frisbee_candidates(
        frame: np.ndarray, detections: list[dict[str, Any]]
    ) -> int:
        height, width = frame.shape[:2]
        frisbee_count = 0
        orange_bgr = (0, 165, 255)

        for detection in detections:
            if str(detection["class_name"]).strip().lower() != "frisbee":
                continue

            x1, y1, x2, y2 = map(int, detection["bbox"])
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            frisbee_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), orange_bgr, 2)
            cv2.putText(
                frame,
                f"frisbee {float(detection['confidence']):.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                orange_bgr,
                2,
            )

        return frisbee_count

    @staticmethod
    def _detect_white_oval_candidates(
        frame: np.ndarray,
        excluded_detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find bright, compact ovals as unverified frisbee-like candidates."""
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 190], dtype=np.uint8),
            np.array([180, 70, 255], dtype=np.uint8),
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = max(16.0, width * height * 0.000015)
        max_area = width * height * 0.02
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area or len(contour) < 5:
                continue

            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 5 or box_height < 5:
                continue
            aspect_ratio = max(box_width, box_height) / min(
                box_width, box_height
            )
            fill_ratio = area / (box_width * box_height)
            if aspect_ratio > 3.2 or fill_ratio < 0.48:
                continue

            overlaps_detection = False
            candidate_box = (x, y, x + box_width, y + box_height)
            for detection in excluded_detections:
                px1, py1, px2, py2 = map(int, detection["bbox"])
                intersection_width = max(
                    0, min(candidate_box[2], px2) - max(candidate_box[0], px1)
                )
                intersection_height = max(
                    0, min(candidate_box[3], py2) - max(candidate_box[1], py1)
                )
                intersection = intersection_width * intersection_height
                if intersection / area >= 0.6:
                    overlaps_detection = True
                    break
            if overlaps_detection:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4.0 * np.pi * area / (perimeter * perimeter)
                if perimeter > 0
                else 0.0
            )
            if circularity < 0.45:
                continue

            candidates.append(
                {
                    "class_name": "frisbee-like",
                    "track_id": None,
                    "confidence": None,
                    "shape_score": round(
                        min(1.0, (fill_ratio + circularity) / 2.0), 2
                    ),
                    "bbox": [
                        float(x),
                        float(y),
                        float(x + box_width),
                        float(y + box_height),
                    ],
                    "status": "White oval heuristic; not model-confirmed",
                }
            )

        candidates.sort(key=lambda candidate: candidate["shape_score"], reverse=True)
        return candidates[:20]

    def process_frame(self, jpeg_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
        if not jpeg_bytes or len(jpeg_bytes) > MAX_FRAME_BYTES:
            raise ValueError("Frame is empty or exceeds the 8 MB limit.")

        encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode the webcam frame.")

        with self._inference_lock:
            with self._state_lock:
                if self._state["status"] != "running":
                    raise RuntimeError("Start tracking before sending frames.")
                inference = self._inference
                adapter = self._adapter
                tracker = self._tracker

            if inference is None or adapter is None or tracker is None:
                raise RuntimeError("Detector initialization is incomplete.")

            start_time = time.perf_counter()
            detections = inference.infer_single(frame)
            boxes, scores, classes, _ = adapter.format_for_deepsort(
                detections, frame.shape
            )
            class_names = {
                int(detection["class_id"]): str(detection["class_name"])
                for detection in detections
            }
            person_indices = np.asarray(
                [
                    index
                    for index, class_id in enumerate(classes)
                    if class_names.get(int(class_id), "").strip().lower()
                    == "person"
                ],
                dtype=np.intp,
            )
            boxes = boxes[person_indices]
            scores = scores[person_indices]
            classes = classes[person_indices]

            track_detections = [
                Detection(
                    box,
                    score,
                    "person",
                    self._appearance_feature(frame, box),
                )
                for box, score in zip(boxes, scores)
            ]
            if track_detections:
                boxes_for_nms = np.asarray(
                    [detection.tlwh for detection in track_detections]
                )
                scores_for_nms = np.asarray(
                    [detection.confidence for detection in track_detections]
                )
                indices = preprocessing.non_max_suppression(
                    boxes_for_nms,
                    np.full(len(track_detections), "person"),
                    1.0,
                    scores_for_nms,
                )
                track_detections = [
                    track_detections[index] for index in indices
                ]

            tracker.predict()
            tracker.update(track_detections)

            visible_players = 0
            player_analysis = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = [round(float(value), 1) for value in track.to_tlbr()]
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 80), 2)
                cv2.putText(
                    frame,
                    f"player-{track.track_id} {float(track.confidence or 0.0):.2f}",
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (50, 220, 80),
                    2,
                )
                visible_players += 1
                player_analysis.append(
                    {
                        "class_name": "person",
                        "track_id": int(track.track_id),
                        "confidence": round(float(track.confidence or 0.0), 3),
                        "bbox": bbox,
                        "status": (
                            "Predicted"
                            if track.time_since_update
                            else "Confirmed player track"
                        ),
                    }
                )

            white_oval_candidates = self._detect_white_oval_candidates(
                frame,
                [
                    detection
                    for detection in detections
                    if str(detection["class_name"]).strip().lower()
                    in {"person", "frisbee", "sports ball"}
                ],
            )
            frisbee_count = self._draw_frisbee_candidates(frame, detections)
            for candidate in white_oval_candidates:
                x1, y1, x2, y2 = map(int, candidate["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    "frisbee-like? white oval",
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    2,
                )
            candidate_analysis = []
            for detection in detections:
                class_name = str(detection["class_name"]).strip().lower()
                if class_name not in {"frisbee", "sports ball"}:
                    continue
                candidate_analysis.append(
                    {
                        "class_name": class_name,
                        "track_id": None,
                        "confidence": round(float(detection["confidence"]), 3),
                        "bbox": [
                            round(float(value), 1)
                            for value in detection["bbox"]
                        ],
                        "status": (
                            "Frisbee-class prediction"
                            if class_name == "frisbee"
                            else "Unconfirmed ball candidate"
                        ),
                    }
                )
            candidate_analysis.extend(white_oval_candidates)
            sports_ball_count = sum(
                item["class_name"] == "sports ball"
                for item in candidate_analysis
            )
            analysis = {
                "frame_size": {
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                },
                "detections": player_analysis + candidate_analysis,
            }
            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            success, output = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            if not success:
                raise RuntimeError("Could not encode the annotated frame.")

            stats = {
                "fps": round(fps, 1),
                "players": visible_players,
                "frisbees": frisbee_count,
                "white_ovals": len(white_oval_candidates),
                "sports_balls": sports_ball_count,
                "person_confidence": inference.person_confidence,
                "analysis": analysis,
            }
            with self._state_lock:
                self._state.update(stats)
            return output.tobytes(), stats

    def stop(self) -> None:
        self._stop_event.set()
        with self._inference_lock:
            with self._state_lock:
                self._inference = None
                self._adapter = None
                self._tracker = None
                self._state = {
                    "status": "stopped",
                    "error": None,
                    "fps": 0.0,
                    "players": 0,
                    "frisbees": 0,
                    "white_ovals": 0,
                    "input_size": None,
                    "device": None,
                    "person_confidence": None,
                }


SERVICE = TrackingService()


class WebcamRequestHandler(BaseHTTPRequestHandler):
    """Serve the single-origin website and local tracking API."""

    server_version = "UltimatePlayerTracker/1.0"

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
        elif self.path == "/styles.css":
            self._send_file(WEB_ROOT / "styles.css", "text/css; charset=utf-8")
        elif self.path == "/app.js":
            self._send_file(
                WEB_ROOT / "app.js", "text/javascript; charset=utf-8"
            )
        elif self.path == "/api/status":
            self._send_json(200, SERVICE.get_state())
        else:
            self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if self.path == "/api/start":
            self._start_tracking()
        elif self.path == "/api/stop":
            SERVICE.stop()
            self._send_json(200, SERVICE.get_state())
        elif self.path == "/api/frame":
            self._process_frame()
        else:
            self._send_json(404, {"error": "Not found."})

    def _start_tracking(self) -> None:
        try:
            body = self._read_json()
            input_size = int(body.get("input_size", 416))
            device = str(body.get("device", "cpu"))
            person_confidence = float(body.get("person_confidence", 0.4))
            SERVICE.start(input_size, device, person_confidence)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
            return
        except FileNotFoundError as error:
            self._send_json(500, {"error": str(error)})
            return
        except RuntimeError as error:
            self._send_json(409, {"error": str(error)})
            return

        self._send_json(202, SERVICE.get_state())

    def _process_frame(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_FRAME_BYTES:
            self._send_json(413, {"error": "Invalid frame size."})
            return
        try:
            frame, stats = SERVICE.process_frame(self.rfile.read(content_length))
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except RuntimeError as error:
            self._send_json(409, {"error": str(error)})
            return
        except Exception as error:
            LOGGER.exception("Webcam frame processing failed.")
            self._send_json(500, {"error": f"Frame processing failed: {error}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("X-Tracker-Fps", str(stats["fps"]))
        self.send_header("X-Tracker-Players", str(stats["players"]))
        self.send_header("X-Tracker-Frisbees", str(stats["frisbees"]))
        self.send_header("X-Tracker-White-Ovals", str(stats["white_ovals"]))
        self.send_header(
            "X-Tracker-Sports-Balls", str(stats["sports_balls"])
        )
        self.send_header(
            "X-Tracker-Analysis",
            json.dumps(stats["analysis"], separators=(",", ":")),
        )
        self.end_headers()
        self.wfile.write(frame)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 64 * 1024:
            raise ValueError("Request body is empty or too large.")
        value = json.loads(self.rfile.read(content_length))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            LOGGER.exception("Could not read website asset %s.", path)
            self._send_json(500, {"error": "Could not load the website asset."})
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)


class LocalWebcamServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local webcam tracker browser interface."
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    address = ("127.0.0.1", args.port)
    try:
        server = LocalWebcamServer(address, WebcamRequestHandler)
    except OSError as error:
        raise SystemExit(
            f"Could not start local website at http://{address[0]}:{address[1]}: "
            f"{error}"
        ) from error

    url = f"http://{address[0]}:{address[1]}"
    print(f"Webcam tracker website: {url}")
    print("The server listens only on this computer. Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local webcam website.")
    finally:
        SERVICE.stop()
        server.server_close()


if __name__ == "__main__":
    main()

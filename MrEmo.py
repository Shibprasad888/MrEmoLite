"""
All the Files has been merged into this single file.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Protocol, Sequence, Tuple

# =============================================================================
# Dependency Checks
# =============================================================================

_ESTIMATING = "Estimating..."


def _require_import(module: str, pip_name: str) -> Any:
    try:
        return __import__(module)
    except ImportError:
        print(f"Missing dependency:\n{pip_name}\n\nRun:\npython3 -m pip install {pip_name}")
        sys.exit(1)


cv2 = _require_import("cv2", "opencv-python")
np = _require_import("numpy", "numpy")
_require_import("tensorflow", "tensorflow")
_require_import("deepface", "deepface")

_psutil_mod: Any = None
try:
    _psutil_mod = __import__("psutil")
except ImportError:
    pass

# =============================================================================
# Configuration & Constants
# =============================================================================

DEBUG: bool = False
DEBUG_MODE: bool = DEBUG  # Set True to log pipeline stages (face detect, emotion inference, etc.)
CAMERA_INDEX: int = 0
CAMERA_TIMEOUT_SEC: float = 10.0
CAMERA_WAITING_SEC: float = 5.0
TARGET_FPS: float = 24.0
TARGET_FRAME_TIME_SEC: float = 1.0 / TARGET_FPS  # 41.67 ms
INFERENCE_MAX_WIDTH: int = 640
EMOTION_INTERVAL: int = 6
HEAD_POSE_INTERVAL: int = 4
EYE_INTERVAL: int = 3
BLINK_INTERVAL: int = 1
SMILE_INTERVAL: int = 5
AGE_INTERVAL_SEC: float = 5.0
GENDER_INTERVAL_SEC: float = 5.0
FALLBACK_DETECT_INTERVAL: int = 15
EMOTION_HISTORY_SIZE: int = 5
DASHBOARD_REFRESH_SEC: float = 1.0
MAX_READ_FAILURES: int = 30
TRACK_GRACE_PERIOD_SEC: float = 3.0
TRACK_MAX_DISTANCE: float = 120.0
BBOX_SMOOTH_ALPHA: float = 0.38
UI_BBOX_SMOOTH_ALPHA: float = 0.45
UI_CONF_SMOOTH_ALPHA: float = 0.32
WINDOW_NAME: str = "Mr Emo"
DETECTOR_BACKEND: str = "retinaface"
APP_NAME: str = "Mr Emo"
APP_VERSION: str = "2.0.0"

EMOTION_CONFIDENCE_THRESHOLD: float = 35.0
GENDER_CONFIDENCE_THRESHOLD: float = 55.0
SMILE_THRESHOLD: float = 0.45
EAR_THRESHOLD: float = 0.21

BoundingBox = Tuple[int, int, int, int]

EMOTION_KEYS: Tuple[str, ...] = (
    "happy",
    "neutral",
    "sad",
    "fear",
    "angry",
    "surprise",
    "disgust",
)

REQUIRED_WEIGHT_FILES: Tuple[str, ...] = (
    "facial_expression_model_weights.h5",
    "age_model_weights.h5",
    "gender_model_weights.h5",
)

USER_NAME: str = "Shiv"

WEIGHT_DOWNLOAD_URLS: Dict[str, str] = {
    "facial_expression_model_weights.h5": (
        "https://github.com/serengil/deepface_models/releases/download/v1.0/"
        "facial_expression_model_weights.h5"
    ),
    "age_model_weights.h5": (
        "https://github.com/serengil/deepface_models/releases/download/v1.0/"
        "age_model_weights.h5"
    ),
    "gender_model_weights.h5": (
        "https://github.com/serengil/deepface_models/releases/download/v1.0/"
        "gender_model_weights.h5"
    ),
}

# BGR palette
PANEL_BG = (28, 28, 32)
PANEL_BG_ALT = (22, 22, 26)
TEXT_PRIMARY = (245, 245, 245)
TEXT_SECONDARY = (170, 170, 180)
TEXT_MUTED = (120, 120, 130)
SHADOW = (0, 0, 0)
STATUS_OK = (90, 220, 120)
STATUS_OFF = (100, 100, 110)
BAR_TRACK = (45, 45, 52)

EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "happy": (80, 220, 100),
    "sad": (240, 130, 70),
    "angry": (80, 80, 240),
    "fear": (70, 150, 255),
    "neutral": (230, 230, 230),
    "surprise": (80, 230, 255),
    "disgust": (200, 80, 200),
    "unknown": (140, 140, 150),
}

EMOTION_EMOJI: Dict[str, str] = {
    "happy": ":)",
    "sad": ":(",
    "angry": "",
    "fear": "",
    "neutral": "",
    "surprise": "",
    "disgust": "",
    "unknown": "",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_ALPHA = 0.76
CORNER_RADIUS = 10
FACE_BORDER_RADIUS = 8
FACE_BORDER_THICKNESS = 2

# MediaPipe Face Mesh landmark indices
_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 263
_RIGHT_EYE_OUTER = 33
_LEFT_EYE_TOP = 386
_LEFT_EYE_BOTTOM = 374
_RIGHT_EYE_TOP = 159
_RIGHT_EYE_BOTTOM = 145
_LEFT_MOUTH = 291
_RIGHT_MOUTH = 61
_UPPER_LIP = 13
_LOWER_LIP = 14

_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


# =============================================================================
# Logging
# =============================================================================

def setup_logging() -> logging.Logger:
    level = logging.DEBUG if DEBUG else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    if not DEBUG:
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("deepface").setLevel(logging.WARNING)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return logging.getLogger("emotion_detection")


logger = setup_logging()


def debug_log(message: str) -> None:
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")


def normalize_score_percent(scores: Dict[str, float]) -> Dict[str, float]:
    """DeepFace Keras outputs are 0–1; UI thresholds expect 0–100."""
    if not scores:
        return scores
    if max(scores.values()) <= 1.0:
        return {key: value * 100.0 for key, value in scores.items()}
    return scores


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FaceAnalysisResult:
    id: int
    bounding_box: BoundingBox
    timestamp: float
    emotion: str = "neutral"
    confidence: float = 0.0
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    age: Optional[int] = None
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    orientation: str = "Looking Center"
    left_eye: str = _ESTIMATING
    right_eye: str = _ESTIMATING
    eyes_state: str = _ESTIMATING
    blink_count: int = 0
    smiling: bool = False
    smile_confidence: float = 0.0
    quality_score: float = 0.0
    quality_label: str = "Medium"
    eyes_closed: bool = False


@dataclass
class LandmarkSet:
    points: Any
    bbox: BoundingBox


@dataclass
class TrackedFace:
    track_id: int
    bounding_box: BoundingBox
    centroid: Tuple[float, float]
    last_seen: float


@dataclass
class AnalysisSmoothingState:
    histories: Dict[int, Deque[str]] = field(default_factory=dict)
    scalar_cache: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class AppState:
    faces: List[FaceAnalysisResult] = field(default_factory=list)
    last_dashboard_time: float = 0.0
    dashboard_terminal_lines: int = 0


@dataclass
class PerformanceMetrics:
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    render_ms: float = 0.0
    fps: float = 0.0
    avg_fps: float = 0.0
    target_fps: float = TARGET_FPS
    frame_count: int = 0


@dataclass
class TrackScheduleState:
    last_emotion_frame: int = -999
    last_pose_frame: int = -999
    last_eye_frame: int = -999
    last_smile_frame: int = -999
    last_age_time: float = 0.0
    last_gender_time: float = 0.0


@dataclass
class FramePacket:
    frame: Any
    frame_id: int
    capture_ms: float
    timestamp: float


@dataclass
class InferencePacket:
    face_updates: List[FaceAnalysisResult]
    fallback_bboxes: List[BoundingBox]
    schedule_updates: Dict[int, TrackScheduleState]
    success: bool
    inference_ms: float
    frame_id: int
    timestamp: float


@dataclass
class LiveDashboardSnapshot:
    fps: float = 0.0
    avg_fps: float = 0.0
    current_faces: int = 0
    total_tracked: int = 0
    inference_ms: float = 0.0
    render_ms: float = 0.0
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    target_fps: float = TARGET_FPS
    emotion_distribution: Dict[str, float] = field(default_factory=dict)
    emotion_counts: Dict[str, int] = field(default_factory=dict)
    camera_connected: bool = False
    ai_ready: bool = False
    tracking_running: bool = False


class FaceRenderable(Protocol):
    id: int
    emotion: str
    confidence: float
    bounding_box: BoundingBox
    age: Optional[int]
    gender: Optional[str]
    gender_confidence: Optional[float]
    orientation: str
    yaw: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    smiling: bool
    smile_confidence: float
    eyes_state: str
    left_eye: str
    right_eye: str
    blink_count: int
    quality_score: float
    quality_label: str
    emotion_scores: Dict[str, float]


# =============================================================================
# Utilities
# =============================================================================

def get_deepface_home() -> Path:
    return Path(os.environ.get("DEEPFACE_HOME", str(Path.home())))


def get_weights_dir() -> Path:
    return get_deepface_home() / ".deepface" / "weights"


def startup_log(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def check_dependencies() -> bool:
    """Verify optional dependencies and report availability."""
    missing: List[str] = []
    if _psutil_mod is None:
        missing.append("psutil (CPU/memory metrics)")
    if missing:
        startup_log("WARNING", f"Optional: {', '.join(missing)} — pip install {' '.join(m.split()[0] for m in missing)}")
    return True


def display_label(value: Optional[str], *, estimating: bool = False) -> str:
    if value is None or (isinstance(value, str) and value.lower() in {"unknown", "n/a", "????"}):
        return _ESTIMATING
    if estimating:
        return _ESTIMATING
    return value


def display_age(age: Optional[int]) -> str:
    return str(age) if age is not None else _ESTIMATING


def display_emotion(emotion: str, confidence: float, scores: Optional[Dict[str, float]] = None) -> str:
    if scores and any(value > 0 for value in scores.values()):
        label = emotion if emotion and emotion.lower() not in {"", "unknown"} else max(scores.items(), key=lambda item: item[1])[0]
        return label.title()
    if confidence > 0 and emotion and emotion.lower() not in {"", "unknown", "neutral"}:
        return emotion.title()
    return _ESTIMATING


def read_yes_no() -> bool:
    while True:
        try:
            answer = input().strip().lower()
        except EOFError:
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter Y or N.")


def _format_progress_bar(percent: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, percent))
    filled = int(width * pct / 100.0)
    return f"{'█' * filled}{'░' * (width - filled)}"


def _render_download_progress(
    filename: str,
    downloaded: int,
    total: int,
    start_time: float,
    *,
    first_draw: bool,
) -> None:
    elapsed = max(time.time() - start_time, 0.001)
    speed = downloaded / elapsed
    if total > 0:
        percent = downloaded * 100.0 / total
        remaining_bytes = max(total - downloaded, 0)
    else:
        percent = 0.0
        remaining_bytes = 0

    remaining_mb = remaining_bytes / (1024 * 1024)
    speed_mb = speed / (1024 * 1024)
    bar = _format_progress_bar(percent)

    if sys.stdout.isatty():
        if first_draw:
            print(f"Downloading {filename}\n")
        progress_lines = (
            f"{bar} {percent:.0f}%\n\n"
            f"Remaining: {remaining_mb:.1f} MB\n\n"
            f"Speed: {speed_mb:.1f} MB/s"
        )
        if first_draw:
            print(progress_lines)
        else:
            sys.stdout.write(f"\033[4A\033[J{progress_lines}")
            sys.stdout.flush()
    elif first_draw or int(percent) % 5 == 0 or percent >= 100:
        print(
            f"Downloading {filename} — {percent:.0f}% "
            f"(remaining {remaining_mb:.1f} MB, {speed_mb:.1f} MB/s)"
        )


def download_weight_file(filename: str, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    for stale in destination.parent.glob(f"{filename}*.part"):
        try:
            stale.unlink()
        except OSError:
            pass

    request = urllib.request.Request(url, headers={"User-Agent": f"MrEmo/{APP_VERSION}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            start_time = time.time()
            first_draw = True
            chunk_size = 256 * 1024

            with open(temp_path, "wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    _render_download_progress(
                        filename,
                        downloaded,
                        total,
                        start_time,
                        first_draw=first_draw,
                    )
                    first_draw = False
    except urllib.error.URLError as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {filename}: {exc}") from exc

    if sys.stdout.isatty():
        sys.stdout.write("\n")
        sys.stdout.flush()

    temp_path.replace(destination)


def downscale_for_inference(frame: Any, max_width: int = INFERENCE_MAX_WIDTH) -> Tuple[Any, float]:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / float(width)
    new_w = max_width
    new_h = max(1, int(height * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return small, scale


def to_inference_bbox(bbox: BoundingBox, scale: float) -> BoundingBox:
    if scale == 1.0:
        return bbox
    x, y, w, h = bbox
    return int(x * scale), int(y * scale), max(1, int(w * scale)), max(1, int(h * scale))


def scale_bbox(bbox: BoundingBox, scale: float) -> BoundingBox:
    if scale == 1.0:
        return bbox
    x, y, w, h = bbox
    inv = 1.0 / scale
    return int(x * inv), int(y * inv), max(1, int(w * inv)), max(1, int(h * inv))


def sanitize_bbox(bbox: BoundingBox, frame_shape: Tuple[int, ...]) -> Optional[BoundingBox]:
    """Clamp a bounding box to frame bounds; return None if invalid."""
    if not frame_shape or len(frame_shape) < 2:
        return None
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        return None

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def smooth_bbox(prev: Optional[BoundingBox], new: BoundingBox, alpha: float = BBOX_SMOOTH_ALPHA) -> BoundingBox:
    if prev is None:
        return new
    return tuple(int(alpha * n + (1.0 - alpha) * p) for p, n in zip(prev, new))  # type: ignore[return-value]


def smooth_scalar(prev: Optional[float], new: Optional[float], alpha: float = 0.25) -> Optional[float]:
    if new is None:
        return prev
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


def majority_emotion(history: Deque[str]) -> str:
    if not history:
        return "neutral"
    return Counter(history).most_common(1)[0][0]


def get_point(landmarks: LandmarkSet, index: int) -> Tuple[float, float]:
    pt = landmarks.points[index]
    return float(pt[0]), float(pt[1])


def centroid(box: BoundingBox) -> Tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def centroid_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ax, ay = centroid(box_a)
    bx, by = centroid(box_b)
    return math.hypot(ax - bx, ay - by)


def crop_face(frame: Any, bbox: BoundingBox) -> Any:
    x, y, w, h = bbox
    height, width = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        return np.empty(0, dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def align_face_crop(frame: Any, landmarks: LandmarkSet, bbox: BoundingBox) -> Any:
    """Align face using eye line for improved emotion inference."""
    x, y, w, h = bbox
    height, width = frame.shape[:2]
    pad = int(max(w, h) * 0.15)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return crop

    left = np.array(get_point(landmarks, _LEFT_EYE_OUTER)) - np.array([x1, y1])
    right = np.array(get_point(landmarks, _RIGHT_EYE_OUTER)) - np.array([x1, y1])
    center = ((left + right) / 2.0).astype(np.float32)
    d_y = right[1] - left[1]
    d_x = right[0] - left[0]
    angle = math.degrees(math.atan2(d_y, d_x))
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    aligned = cv2.warpAffine(
        crop,
        matrix,
        (crop.shape[1], crop.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def read_cpu_percent() -> Optional[float]:
    if _psutil_mod is None:
        return None
    return _psutil_mod.cpu_percent(interval=None)


def read_memory_percent() -> Optional[float]:
    if _psutil_mod is None:
        return None
    return _psutil_mod.virtual_memory().percent


# =============================================================================
# Model Initialization & Caching
# =============================================================================

class ModelCache:
    """Preloads DeepFace models once; skips download when weights exist locally."""

    def __init__(self) -> None:
        self.weights_dir = get_weights_dir()
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self._emotion_model = None
        self._age_model = None
        self._gender_model = None
        self._ready = False
        self._detector_warmed = False
        self.user_declined_download = False

    def _weight_ready(self, name: str) -> bool:
        path = self.weights_dir / name
        return path.is_file() and path.stat().st_size > 0

    def _missing_weights(self) -> List[str]:
        return [name for name in REQUIRED_WEIGHT_FILES if not self._weight_ready(name)]

    def _ensure_cached_weights(self) -> bool:
        missing = self._missing_weights()
        if not missing:
            startup_log("INFO", "Checking local model cache...")
            startup_log("SUCCESS", "All required models found.")
            return True

        startup_log("INFO", "Initializing AI models...")
        startup_log("INFO", "Checking local model cache...")
        print("-" * 38)
        startup_log("INFO", "Missing model detected:")
        for name in missing:
            print(name)
        print("This model will be downloaded once.")
        print("Download size may take a few minutes depending on your internet connection.")
        print("Continue? (Y/N):")
        print("-" * 38)

        if not read_yes_no():
            self.user_declined_download = True
            return False

        try:
            for name in missing:
                url = WEIGHT_DOWNLOAD_URLS[name]
                download_weight_file(name, url, self.weights_dir / name)
            startup_log("SUCCESS", "Model downloaded successfully.")
            print("Future launches will use the cached model.")
        except Exception as exc:
            logger.error("Model download failed: %s", exc)
            return False

        return True

    def _load_models(self) -> bool:
        logger.info("Loading models from %s", self.weights_dir)

        try:
            from deepface import DeepFace

            self._emotion_model = DeepFace.build_model(task="facial_attribute", model_name="Emotion")
            self._age_model = DeepFace.build_model(task="facial_attribute", model_name="Age")
            self._gender_model = DeepFace.build_model(task="facial_attribute", model_name="Gender")
            self._warm_detector()
            self._ready = True
            startup_log("SUCCESS", f"{USER_NAME}, AI models loaded successfully.")
            logger.info("Models loaded successfully.")
            return True
        except ImportError as exc:
            logger.error("DeepFace not installed: %s", exc)
        except Exception as exc:
            logger.error("Model loading failed: %s", exc)
        return False

    def prepare_and_load(self) -> bool:
        if not self._ensure_cached_weights():
            return False
        return self._load_models()

    def initialize(self) -> bool:
        return self.prepare_and_load()

    def _warm_detector(self) -> None:
        if self._detector_warmed:
            return
        try:
            from deepface import DeepFace

            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            DeepFace.analyze(
                dummy,
                actions=["emotion"],
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True,
            )
            self._detector_warmed = True
            logger.info("Detector backend '%s' warmed up.", DETECTOR_BACKEND)
        except Exception as exc:
            logger.warning("Detector warm-up skipped: %s", str(exc).splitlines()[0])

    @property
    def ready(self) -> bool:
        return self._ready

    def _prepare_crop_batch(self, aligned_bgr: Any) -> Optional[Any]:
        if aligned_bgr.size == 0:
            return None
        try:
            from deepface.modules import preprocessing

            return preprocessing.resize_image(img=aligned_bgr, target_size=(224, 224))
        except Exception as exc:
            if DEBUG_MODE:
                debug_log(f"Crop resize failed: {exc}")
            logger.debug("Crop resize failed: %s", exc)
            return None

    def predict_emotion_on_crop(self, aligned_bgr: Any) -> Optional[Dict[str, float]]:
        if self._emotion_model is None or aligned_bgr.size == 0:
            return None
        batch = self._prepare_crop_batch(aligned_bgr)
        if batch is None:
            return None
        try:
            from deepface.models.demography import Emotion as EmotionModule

            preds = self._emotion_model.predict(batch)
            if hasattr(preds, "ndim") and preds.ndim > 1:
                preds = preds[0]
            total = float(np.sum(preds))
            if total <= 0:
                return None
            return {
                EmotionModule.labels[i]: 100.0 * float(preds[i]) / total
                for i in range(len(EmotionModule.labels))
            }
        except Exception as exc:
            if DEBUG_MODE:
                debug_log(f"Emotion predict failed: {exc}")
            logger.debug("Emotion predict failed: %s", exc)
            return None

    def predict_age_on_crop(self, aligned_bgr: Any) -> Optional[int]:
        if self._age_model is None or aligned_bgr.size == 0:
            return None
        batch = self._prepare_crop_batch(aligned_bgr)
        if batch is None:
            return None
        try:
            age_val = self._age_model.predict(batch)
            return int(round(float(age_val)))
        except Exception as exc:
            if DEBUG_MODE:
                debug_log(f"Age predict failed: {exc}")
            logger.debug("Age predict failed: %s", exc)
            return None

    def predict_gender_on_crop(self, aligned_bgr: Any) -> Tuple[Optional[str], Optional[float]]:
        if self._gender_model is None or aligned_bgr.size == 0:
            return None, None
        batch = self._prepare_crop_batch(aligned_bgr)
        if batch is None:
            return None, None
        try:
            from deepface.models.demography import Gender as GenderModule

            preds = self._gender_model.predict(batch)
            if hasattr(preds, "ndim") and preds.ndim > 1:
                preds = preds[0]
            scores = {
                GenderModule.labels[i].lower(): 100.0 * float(preds[i])
                for i in range(min(len(GenderModule.labels), len(preds)))
            }
            if not scores:
                return None, None
            top_key = max(scores.items(), key=lambda item: item[1])[0]
            confidence = scores[top_key]
            if confidence < GENDER_CONFIDENCE_THRESHOLD:
                return None, confidence
            label = "Female" if "woman" in top_key else "Male"
            return label, confidence
        except Exception as exc:
            if DEBUG_MODE:
                debug_log(f"Gender predict failed: {exc}")
            logger.debug("Gender predict failed: %s", exc)
            return None, None


# =============================================================================
# Performance Tracking
# =============================================================================

class PerformanceTracker:
    def __init__(self) -> None:
        self.metrics = PerformanceMetrics()
        self._last_tick: Optional[float] = None
        self._frame_times: Deque[float] = deque(maxlen=240)

    def tick(self, capture_ms: float, render_ms: float, inference_ms: float) -> PerformanceMetrics:
        now = time.perf_counter()
        m = self.metrics

        if self._last_tick is not None:
            dt = now - self._last_tick
            if dt > 0:
                m.fps = min(1.0 / dt, TARGET_FPS * 1.5)

        self._last_tick = now
        self._frame_times.append(now)

        cutoff = now - 1.0
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()

        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            m.avg_fps = min((len(self._frame_times) - 1) / span, TARGET_FPS * 1.5) if span > 0 else m.fps
        else:
            m.avg_fps = m.fps

        m.target_fps = TARGET_FPS

        m.capture_ms = capture_ms
        m.render_ms = render_ms
        m.inference_ms = inference_ms
        m.frame_count += 1
        return m


# =============================================================================
# Pipeline — Threaded Capture & Inference
# =============================================================================

class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: Optional[FramePacket] = None

    def publish(self, packet: FramePacket) -> None:
        with self._lock:
            self._packet = packet

    def consume(self) -> Optional[FramePacket]:
        with self._lock:
            return self._packet


class LatestInferenceResult:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: Optional[InferencePacket] = None
        self._last_consumed_id: int = -1

    def publish(self, packet: InferencePacket) -> None:
        with self._lock:
            self._packet = packet

    def poll(self) -> Optional[InferencePacket]:
        with self._lock:
            if self._packet is None or self._packet.frame_id <= self._last_consumed_id:
                return None
            self._last_consumed_id = self._packet.frame_id
            return self._packet

    def latest(self) -> Optional[InferencePacket]:
        with self._lock:
            return self._packet


def configure_camera_for_performance(cap: Any) -> None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


class CameraCaptureThread(threading.Thread):
    daemon = True

    def __init__(self, cap: Any, buffer: LatestFrameBuffer, max_failures: int = 30) -> None:
        super().__init__(name="CameraCapture")
        self._cap = cap
        self._buffer = buffer
        self._max_failures = max_failures
        self._running = threading.Event()
        self._running.set()
        self._frame_id = 0
        self._failures = 0
        self._last_capture_ms = 0.0
        self._frame_buffer: Optional[Any] = None

    @property
    def last_capture_ms(self) -> float:
        return self._last_capture_ms

    @property
    def failures(self) -> int:
        return self._failures

    def stop(self) -> None:
        self._running.clear()

    def run(self) -> None:
        while self._running.is_set():
            start = time.perf_counter()
            try:
                ret, frame = self._cap.read()
            except cv2.error as exc:
                logger.debug("Capture error: %s", exc)
                self._failures += 1
                time.sleep(0.01)
                continue

            self._last_capture_ms = (time.perf_counter() - start) * 1000.0

            if not ret or frame is None:
                self._failures += 1
                if self._failures >= self._max_failures:
                    logger.error("Capture thread: too many read failures.")
                    self._running.clear()
                time.sleep(0.005)
                continue

            self._failures = 0
            self._frame_id += 1

            if self._frame_buffer is None or self._frame_buffer.shape != frame.shape:
                self._frame_buffer = np.empty_like(frame)
            np.copyto(self._frame_buffer, frame)

            self._buffer.publish(
                FramePacket(
                    frame=self._frame_buffer,
                    frame_id=self._frame_id,
                    capture_ms=self._last_capture_ms,
                    timestamp=time.time(),
                )
            )


class InferenceWorker(threading.Thread):
    daemon = True

    def __init__(
        self,
        analyzer: CompositeFaceAnalyzer,
        results: LatestInferenceResult,
    ) -> None:
        super().__init__(name="InferenceWorker")
        self._analyzer = analyzer
        self._results = results
        self._queue: queue.Queue[
            Tuple[Any, float, int, float, List[TrackedFace], List[LandmarkSet], bool, Dict[int, FaceAnalysisResult], Dict[int, TrackScheduleState]]
        ] = queue.Queue(maxsize=1)
        self._running = threading.Event()
        self._running.set()

    def submit(
        self,
        frame: Any,
        frame_id: int,
        timestamp: float,
        tracked_faces: List[TrackedFace],
        landmark_sets: List[LandmarkSet],
        run_fallback: bool,
        face_cache: Dict[int, FaceAnalysisResult],
        schedules: Dict[int, TrackScheduleState],
    ) -> None:
        small, infer_scale = downscale_for_inference(frame)
        cache_copy = {
            key: FaceAnalysisResult(
                id=value.id,
                bounding_box=value.bounding_box,
                timestamp=value.timestamp,
                emotion=value.emotion,
                confidence=value.confidence,
                emotion_scores=dict(value.emotion_scores),
                age=value.age,
                gender=value.gender,
                gender_confidence=value.gender_confidence,
                yaw=value.yaw,
                pitch=value.pitch,
                roll=value.roll,
                orientation=value.orientation,
                left_eye=value.left_eye,
                right_eye=value.right_eye,
                eyes_state=value.eyes_state,
                blink_count=value.blink_count,
                smiling=value.smiling,
                smile_confidence=value.smile_confidence,
                quality_score=value.quality_score,
                quality_label=value.quality_label,
                eyes_closed=value.eyes_closed,
            )
            for key, value in face_cache.items()
        }
        schedule_copy = {
            key: TrackScheduleState(
                last_emotion_frame=value.last_emotion_frame,
                last_pose_frame=value.last_pose_frame,
                last_eye_frame=value.last_eye_frame,
                last_smile_frame=value.last_smile_frame,
                last_age_time=value.last_age_time,
                last_gender_time=value.last_gender_time,
            )
            for key, value in schedules.items()
        }
        payload = (
            small, infer_scale, frame_id, timestamp, list(tracked_faces),
            list(landmark_sets), run_fallback, cache_copy, schedule_copy,
        )
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._running.clear()
        try:
            self._queue.put_nowait((
                np.empty(0), 1.0, -1, 0.0, [], [], False, {}, {},
            ))
        except queue.Full:
            pass

    def run(self) -> None:
        while self._running.is_set():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame, scale, frame_id, timestamp, tracked, landmark_sets, run_fallback, cache, schedules = payload
            if frame_id < 0:
                break

            start = time.perf_counter()
            try:
                updates, fallback, schedule_updates, success = analyze_tracks_slow(
                    frame, scale, tracked, landmark_sets, frame_id, timestamp,
                    schedules, cache, self._analyzer, run_fallback=run_fallback,
                )
            except Exception as exc:
                logger.warning("Inference worker error: %s", exc)
                updates, fallback, schedule_updates, success = [], [], {}, False

            inference_ms = (time.perf_counter() - start) * 1000.0
            self._results.publish(
                InferencePacket(
                    face_updates=updates,
                    fallback_bboxes=fallback,
                    schedule_updates=schedule_updates,
                    success=success,
                    inference_ms=inference_ms,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )


class DisplayBuffer:
    def __init__(self) -> None:
        self._buffer: Optional[Any] = None

    def prepare(self, frame: Any) -> Any:
        if self._buffer is None or self._buffer.shape != frame.shape:
            self._buffer = np.empty_like(frame)
        np.copyto(self._buffer, frame)
        return self._buffer


# =============================================================================
# Camera Startup
# =============================================================================

def initialize_camera(
    index: int = CAMERA_INDEX,
    timeout_sec: float = CAMERA_TIMEOUT_SEC,
    waiting_sec: float = CAMERA_WAITING_SEC,
) -> Optional[Any]:
    start_time = time.time()
    waiting_logged = False
    permission_warning_logged = False

    while time.time() - start_time < timeout_sec:
        try:
            candidate = cv2.VideoCapture(index)
            if candidate.isOpened():
                ret, frame = candidate.read()
                if ret and frame is not None:
                    configure_camera_for_performance(candidate)
                    startup_log("SUCCESS", "Camera connected.")
                    logger.info("Camera initialized successfully.")
                    return candidate
                candidate.release()
            else:
                candidate.release()
        except cv2.error as exc:
            logger.debug("OpenCV camera error during init: %s", exc)
        except Exception as exc:
            logger.debug("Unexpected camera init error: %s", exc)

        elapsed = time.time() - start_time
        if elapsed >= waiting_sec and not waiting_logged:
            startup_log("INFO", "Waiting for camera permission...")
            waiting_logged = True
        if elapsed >= waiting_sec and not permission_warning_logged:
            startup_log(
                "WARNING",
                f"{USER_NAME}, camera permission is currently restricted.",
            )
            print("Please allow camera access in macOS Privacy & Security.")
            permission_warning_logged = True

        time.sleep(0.1)

    startup_log(
        "ERROR",
        f"{USER_NAME}, the application cannot continue until the required resource is available.",
    )
    return None


def cleanup(cap: Optional[Any]) -> None:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    logger.info("Cleanup complete.")


# =============================================================================
# Face Detection — Landmarks
# =============================================================================

class LandmarkProvider:
    def __init__(self) -> None:
        self._mp_face_mesh = None
        self._use_mediapipe = False
        try:
            mp = __import__("mediapipe")
            face_mesh_module = getattr(mp, "solutions", None)
            face_mesh_cls = None
            if face_mesh_module is not None:
                face_mesh_cls = getattr(face_mesh_module, "face_mesh", None)
                if face_mesh_cls is not None:
                    face_mesh_cls = getattr(face_mesh_cls, "FaceMesh", None)
            if face_mesh_cls is None:
                raise AttributeError("MediaPipe face mesh API is not available in this installation")

            self._mp_face_mesh = face_mesh_cls(
                static_image_mode=False,
                max_num_faces=8,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._use_mediapipe = True
            logger.info("MediaPipe Face Mesh initialized.")
        except Exception as exc:
            self._use_mediapipe = False
            startup_log(
                "WARNING",
                "Optional: mediapipe unavailable in this environment; landmark pose will use estimates without it",
            )
            debug_log(f"MediaPipe initialization skipped: {exc}")

    @property
    def available(self) -> bool:
        return self._use_mediapipe

    def detect(self, frame: Any) -> List[LandmarkSet]:
        if not self._use_mediapipe or self._mp_face_mesh is None:
            return []

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mp_face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []

        sets: List[LandmarkSet] = []
        for face_lm in result.multi_face_landmarks:
            points = np.array(
                [(lm.x * width, lm.y * height) for lm in face_lm.landmark],
                dtype=np.float64,
            )
            xs, ys = points[:, 0], points[:, 1]
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            sets.append(LandmarkSet(points=points, bbox=bbox))
        return sets

    @staticmethod
    def match_landmarks(bbox: BoundingBox, landmark_sets: List[LandmarkSet]) -> Optional[LandmarkSet]:
        if not landmark_sets:
            return None

        best: Optional[LandmarkSet] = None
        best_dist = float("inf")
        bx, by, bw, bh = bbox
        bcx, bcy = bx + bw / 2, by + bh / 2

        for lm_set in landmark_sets:
            lx, ly, lw, lh = lm_set.bbox
            lcx, lcy = lx + lw / 2, ly + lh / 2
            dist = np.hypot(bcx - lcx, bcy - lcy)
            if dist < best_dist:
                best_dist = dist
                best = lm_set

        if best is not None and best_dist > max(bw, bh) * 0.8:
            return None
        return best

    @staticmethod
    def synthetic_landmarks(bbox: BoundingBox) -> LandmarkSet:
        x, y, w, h = bbox
        points = np.zeros((468, 2), dtype=np.float64)
        anchors = {
            _NOSE_TIP: (x + w * 0.50, y + h * 0.55),
            _CHIN: (x + w * 0.50, y + h * 0.92),
            _LEFT_EYE_OUTER: (x + w * 0.72, y + h * 0.38),
            _RIGHT_EYE_OUTER: (x + w * 0.28, y + h * 0.38),
            _LEFT_EYE_TOP: (x + w * 0.72, y + h * 0.33),
            _LEFT_EYE_BOTTOM: (x + w * 0.72, y + h * 0.43),
            _RIGHT_EYE_TOP: (x + w * 0.28, y + h * 0.33),
            _RIGHT_EYE_BOTTOM: (x + w * 0.28, y + h * 0.43),
            _LEFT_MOUTH: (x + w * 0.65, y + h * 0.72),
            _RIGHT_MOUTH: (x + w * 0.35, y + h * 0.72),
            _UPPER_LIP: (x + w * 0.50, y + h * 0.68),
            _LOWER_LIP: (x + w * 0.50, y + h * 0.76),
        }
        for idx, coord in anchors.items():
            points[idx] = coord
        return LandmarkSet(points=points, bbox=bbox)


# =============================================================================
# Emotion Recognition & Analysis Modules
# =============================================================================

class DeepFaceBackend:
    def __init__(self, detector_backend: str = DETECTOR_BACKEND) -> None:
        self.detector_backend = detector_backend

    def analyze(self, frame: Any) -> List[Any]:
        from deepface import DeepFace

        results = DeepFace.analyze(
            frame,
            actions=["emotion", "age", "gender"],
            detector_backend=self.detector_backend,
            enforce_detection=False,
            silent=True,
        )
        if isinstance(results, dict):
            return [results]
        if isinstance(results, list):
            return results
        return [results]

    @staticmethod
    def region(payload: dict) -> Optional[BoundingBox]:
        region = payload.get("region")
        if not region:
            return None
        x, y, w, h = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h


class EmotionRecognizer:
    def __init__(self, model_cache: ModelCache) -> None:
        self._cache = model_cache

    def analyze(
        self,
        frame: Any,
        bbox: BoundingBox,
        landmarks: Optional[LandmarkSet],
        deepface_payload: Optional[dict],
        *,
        track_id: Optional[int] = None,
    ) -> dict:
        scores: Dict[str, float] = {}
        if deepface_payload:
            scores_raw = deepface_payload.get("emotion", {})
            scores = normalize_score_percent(
                {str(k).lower(): float(v) for k, v in scores_raw.items()} if scores_raw else {}
            )

        face_img: Any = np.empty(0, dtype=frame.dtype)
        if landmarks is not None and HeadPoseEstimator._landmarks_reliable(landmarks):
            face_img = align_face_crop(frame, landmarks, bbox)
        if face_img.size == 0:
            face_img = crop_face(frame, bbox)

        if face_img.size > 0:
            if DEBUG_MODE:
                debug_log(f"Emotion inference started — Face ID {track_id}")
            infer_start = time.perf_counter()
            refined = self._cache.predict_emotion_on_crop(face_img)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            if refined:
                for key, value in refined.items():
                    scores[key] = max(scores.get(key, 0.0), value)
                if DEBUG_MODE:
                    dominant = max(scores.items(), key=lambda item: item[1])[0]
                    debug_log(f"Emotion inference completed — Face ID {track_id}")
                    debug_log(f"Inference time: {infer_ms:.0f} ms")
                    debug_log(f"Returned emotions: {scores}")
                    debug_log(f"Dominant emotion: {dominant}")
                    debug_log(f"Confidence: {scores[dominant]:.1f}")
            elif DEBUG_MODE:
                debug_log(f"Emotion inference failed — Face ID {track_id} (no scores returned)")

        if not scores:
            return {"emotion": "neutral", "confidence": 0.0, "emotion_scores": {}}

        scores = normalize_score_percent(scores)
        dominant = max(scores.items(), key=lambda item: item[1])[0]
        confidence = scores[dominant]
        return {"emotion": dominant, "confidence": confidence, "emotion_scores": scores}


class HeadPoseEstimator:
    @staticmethod
    def _landmarks_reliable(landmarks: LandmarkSet) -> bool:
        return int(np.count_nonzero(landmarks.points)) > 24

    @staticmethod
    def analyze(
        frame: Any,
        bbox: BoundingBox,
        landmarks: Optional[LandmarkSet],
        deepface_payload: Optional[dict],
    ) -> dict:
        del deepface_payload, bbox
        if landmarks is None or not HeadPoseEstimator._landmarks_reliable(landmarks):
            return {"yaw": None, "pitch": None, "roll": None}

        lm = landmarks
        height, width = frame.shape[:2]

        image_points = np.array(
            [
                get_point(lm, _NOSE_TIP),
                get_point(lm, _CHIN),
                get_point(lm, _LEFT_EYE_OUTER),
                get_point(lm, _RIGHT_EYE_OUTER),
                get_point(lm, _LEFT_MOUTH),
                get_point(lm, _RIGHT_MOUTH),
            ],
            dtype=np.float64,
        )

        focal = float(width)
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))

        try:
            ok, rotation_vector, _ = cv2.solvePnP(
                _MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return {"yaw": None, "pitch": None, "roll": None}

            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            proj = np.hstack((rotation_mat, np.zeros((3, 1), dtype=np.float64)))
            proj_full = np.vstack((proj, np.array([[0, 0, 0, 1]], dtype=np.float64)))
            euler = cv2.decomposeProjectionMatrix(proj_full)[5]
            vals = [float(v) for v in euler.flatten()[:3]]
            pitch, yaw, roll = vals[0], vals[1], vals[2]
            return {"yaw": yaw, "pitch": pitch, "roll": roll}
        except cv2.error:
            return {"yaw": None, "pitch": None, "roll": None}


class FaceOrientationAnalyzer:
    YAW_THRESHOLD = 12.0
    PITCH_THRESHOLD = 10.0

    @classmethod
    def analyze(
        cls,
        frame: Any,
        bbox: BoundingBox,
        landmarks: Optional[LandmarkSet],
        deepface_payload: Optional[dict],
        pose: Optional[dict] = None,
    ) -> dict:
        del frame, bbox, landmarks, deepface_payload
        yaw = pitch = None
        if pose:
            yaw = pose.get("yaw")
            pitch = pose.get("pitch")

        if yaw is None or pitch is None:
            return {"orientation": _ESTIMATING}

        if yaw <= -cls.YAW_THRESHOLD:
            return {"orientation": "Looking Left"}
        if yaw >= cls.YAW_THRESHOLD:
            return {"orientation": "Looking Right"}
        if pitch <= -cls.PITCH_THRESHOLD:
            return {"orientation": "Looking Up"}
        if pitch >= cls.PITCH_THRESHOLD:
            return {"orientation": "Looking Down"}
        return {"orientation": "Looking Center"}


class EyeAnalyzer:
    @staticmethod
    def _landmarks_reliable(landmarks: LandmarkSet) -> bool:
        return landmarks.points.shape[0] >= 468

    @staticmethod
    def _eye_aspect_ratio(landmarks: LandmarkSet, outer_idx: int, top_idx: int, bottom_idx: int) -> float:
        ox, oy = get_point(landmarks, outer_idx)
        tx, ty = get_point(landmarks, top_idx)
        bx, by = get_point(landmarks, bottom_idx)
        vertical = np.hypot(tx - bx, ty - by)
        horizontal = max(np.hypot(ox - tx, oy - ty), 1.0)
        return float(vertical / horizontal)

    def analyze(
        self,
        frame: Any,
        bbox: BoundingBox,
        landmarks: Optional[LandmarkSet],
        deepface_payload: Optional[dict],
    ) -> dict:
        del frame, deepface_payload
        if landmarks is None or not self._landmarks_reliable(landmarks):
            return {
                "left_eye": _ESTIMATING,
                "right_eye": _ESTIMATING,
                "eyes_state": _ESTIMATING,
                "eyes_closed": False,
            }

        left_ear = self._eye_aspect_ratio(landmarks, _LEFT_EYE_OUTER, _LEFT_EYE_TOP, _LEFT_EYE_BOTTOM)
        right_ear = self._eye_aspect_ratio(landmarks, _RIGHT_EYE_OUTER, _RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM)

        left_open = left_ear >= EAR_THRESHOLD
        right_open = right_ear >= EAR_THRESHOLD

        left_state = "Open" if left_open else "Closed"
        right_state = "Open" if right_open else "Closed"

        if left_open and right_open:
            aggregate = "Open"
        elif not left_open and not right_open:
            aggregate = "Closed"
        else:
            aggregate = "Partial"

        return {
            "left_eye": left_state,
            "right_eye": right_state,
            "eyes_state": aggregate,
            "eyes_closed": not left_open and not right_open,
        }


class SmileDetector:
    @staticmethod
    def analyze(
        frame: Any,
        bbox: BoundingBox,
        landmarks: Optional[LandmarkSet],
        deepface_payload: Optional[dict],
        emotion_scores: Optional[dict] = None,
    ) -> dict:
        del frame, bbox
        happy_score = 0.0
        if emotion_scores:
            happy_score = float(emotion_scores.get("happy", 0.0)) / 100.0
        elif deepface_payload and deepface_payload.get("emotion"):
            happy_score = float(deepface_payload["emotion"].get("happy", 0.0)) / 100.0

        geometry_score = 0.0
        if landmarks is not None:
            lx, ly = get_point(landmarks, _LEFT_MOUTH)
            rx, ry = get_point(landmarks, _RIGHT_MOUTH)
            ux, uy = get_point(landmarks, _UPPER_LIP)
            mouth_width = max(np.hypot(lx - rx, ly - ry), 1.0)
            mouth_center_y = (ly + ry) / 2.0
            corner_lift = max(0.0, (uy - mouth_center_y) / mouth_width)
            lip_gap = max(np.hypot(*np.subtract(get_point(landmarks, _LOWER_LIP), (ux, uy))), 0.0)
            geometry_score = min(1.0, corner_lift * 2.5 + lip_gap / mouth_width * 0.3)

        confidence = min(1.0, happy_score * 0.65 + geometry_score * 0.35)
        smiling = confidence >= SMILE_THRESHOLD
        return {"smiling": smiling, "smile_confidence": confidence * 100.0}


class FaceQualityAnalyzer:
    @staticmethod
    def analyze(
        frame: Any,
        bbox: BoundingBox,
        landmarks: Optional[LandmarkSet],
        deepface_payload: Optional[dict],
    ) -> dict:
        del landmarks, deepface_payload
        x, y, w, h = bbox
        height, width = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            return {"quality_score": 0.0, "quality_label": "Low"}

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        size_score = min(1.0, (w * h) / (width * height) * 20.0)
        sharp_score = min(1.0, sharpness / 180.0)
        score = (sharp_score * 0.7 + size_score * 0.3) * 100.0

        if score >= 65:
            label = "High"
        elif score >= 35:
            label = "Medium"
        else:
            label = "Low"
        return {"quality_score": score, "quality_label": label}


class CompositeFaceAnalyzer:
    """Orchestrates scheduled per-track analysis — fast tasks on main thread, slow tasks on worker."""

    _haar_cascade: Optional[Any] = None

    def __init__(self, model_cache: ModelCache, detector_backend: str = DETECTOR_BACKEND) -> None:
        self._cache = model_cache
        self._deepface = DeepFaceBackend(detector_backend)
        self.landmarks = LandmarkProvider()
        self._emotion = EmotionRecognizer(model_cache)
        self._head_pose = HeadPoseEstimator()
        self._orientation = FaceOrientationAnalyzer()
        self._eyes = EyeAnalyzer()
        self._smile = SmileDetector()
        self._quality = FaceQualityAnalyzer()
        if CompositeFaceAnalyzer._haar_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            CompositeFaceAnalyzer._haar_cascade = cv2.CascadeClassifier(cascade_path)

    @property
    def mediapipe_available(self) -> bool:
        return self.landmarks.available

    @classmethod
    def _detect_haar(cls, frame: Any) -> List[BoundingBox]:
        if cls._haar_cascade is None or cls._haar_cascade.empty():
            return []
        if frame is None or frame.size == 0 or frame.ndim < 2:
            return []

        height, width = frame.shape[:2]
        if height < 24 or width < 24:
            return []

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return []
        if not gray.flags["C_CONTIGUOUS"]:
            gray = np.ascontiguousarray(gray)

        min_dim = min(height, width)
        min_size = max(20, min(40, min_dim // 5))
        try:
            detections = cls._haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_size, min_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except cv2.error as exc:
            if DEBUG_MODE:
                debug_log(f"Haar detection failed: {exc}")
            return []

        bboxes: List[BoundingBox] = []
        for x, y, w, h in detections:
            sanitized = sanitize_bbox((int(x), int(y), int(w), int(h)), frame.shape)
            if sanitized is not None:
                bboxes.append(sanitized)
        return bboxes

    def detect_faces(self, frame: Any) -> Tuple[List[BoundingBox], List[LandmarkSet]]:
        sets = self.landmarks.detect(frame)
        if sets:
            if DEBUG_MODE:
                debug_log(f"Face detected — MediaPipe ({len(sets)} face(s))")
            return [item.bbox for item in sets], sets

        haar_bboxes = self._detect_haar(frame)
        if haar_bboxes:
            if DEBUG_MODE:
                debug_log(f"Face detected — OpenCV Haar ({len(haar_bboxes)} face(s))")
            haar_sets = [LandmarkProvider.synthetic_landmarks(bbox) for bbox in haar_bboxes]
            return haar_bboxes, haar_sets

        if DEBUG_MODE:
            debug_log("Face detection — no faces found this frame")
        return [], []

    def detect_fallback_bboxes(self, frame: Any, scale: float) -> List[BoundingBox]:
        haar_bboxes = self._detect_haar(frame)
        if not haar_bboxes:
            return []
        if scale != 1.0:
            return [scale_bbox(bbox, scale) for bbox in haar_bboxes]
        return haar_bboxes

    @staticmethod
    def _resolve_landmarks(bbox: BoundingBox, landmark_sets: List[LandmarkSet]) -> LandmarkSet:
        matched = LandmarkProvider.match_landmarks(bbox, landmark_sets)
        if matched is not None:
            return matched
        return LandmarkProvider.synthetic_landmarks(bbox)

    def _clone_cached(self, track_id: int, bbox: BoundingBox, timestamp: float, cached: FaceAnalysisResult) -> FaceAnalysisResult:
        return FaceAnalysisResult(
            id=track_id,
            bounding_box=bbox,
            timestamp=timestamp,
            emotion=cached.emotion,
            confidence=cached.confidence,
            emotion_scores=dict(cached.emotion_scores),
            age=cached.age,
            gender=cached.gender,
            gender_confidence=cached.gender_confidence,
            yaw=cached.yaw,
            pitch=cached.pitch,
            roll=cached.roll,
            orientation=cached.orientation,
            left_eye=cached.left_eye,
            right_eye=cached.right_eye,
            eyes_state=cached.eyes_state,
            blink_count=cached.blink_count,
            smiling=cached.smiling,
            smile_confidence=cached.smile_confidence,
            quality_score=cached.quality_score,
            quality_label=cached.quality_label,
            eyes_closed=cached.eyes_closed,
        )

    def update_track_fast(
        self,
        frame: Any,
        track_id: int,
        bbox: BoundingBox,
        landmarks: LandmarkSet,
        frame_id: int,
        timestamp: float,
        schedule: TrackScheduleState,
        cached: FaceAnalysisResult,
    ) -> FaceAnalysisResult:
        result = self._clone_cached(track_id, bbox, timestamp, cached)
        reliable = HeadPoseEstimator._landmarks_reliable(landmarks)

        if reliable:
            eye_data = self._eyes.analyze(frame, bbox, landmarks, None)
            result.eyes_closed = bool(eye_data.get("eyes_closed", False))
            if frame_id - schedule.last_eye_frame >= EYE_INTERVAL:
                result.left_eye = str(eye_data.get("left_eye", _ESTIMATING))
                result.right_eye = str(eye_data.get("right_eye", _ESTIMATING))
                result.eyes_state = str(eye_data.get("eyes_state", _ESTIMATING))
                schedule.last_eye_frame = frame_id

        if reliable and frame_id - schedule.last_pose_frame >= HEAD_POSE_INTERVAL:
            pose_data = self._head_pose.analyze(frame, bbox, landmarks, None)
            orient_data = self._orientation.analyze(frame, bbox, landmarks, None, pose=pose_data)
            result.yaw = pose_data.get("yaw")
            result.pitch = pose_data.get("pitch")
            result.roll = pose_data.get("roll")
            result.orientation = str(orient_data.get("orientation", _ESTIMATING))
            schedule.last_pose_frame = frame_id

        if frame_id - schedule.last_smile_frame >= SMILE_INTERVAL:
            smile_data = self._smile.analyze(
                frame, bbox, landmarks if reliable else None, None,
                emotion_scores=result.emotion_scores,
            )
            result.smiling = bool(smile_data.get("smiling", False))
            result.smile_confidence = float(smile_data.get("smile_confidence", 0.0))
            schedule.last_smile_frame = frame_id

        quality_data = self._quality.analyze(frame, bbox, landmarks if reliable else None, None)
        result.quality_score = float(quality_data.get("quality_score", result.quality_score))
        result.quality_label = str(quality_data.get("quality_label", result.quality_label))

        return result

    def update_tracks_slow(
        self,
        frame: Any,
        scale: float,
        tracked_faces: List[TrackedFace],
        landmark_sets: List[LandmarkSet],
        frame_id: int,
        timestamp: float,
        schedules: Dict[int, TrackScheduleState],
        face_cache: Dict[int, FaceAnalysisResult],
        *,
        run_fallback: bool,
    ) -> Tuple[List[FaceAnalysisResult], List[BoundingBox]]:
        fallback_bboxes: List[BoundingBox] = []
        if run_fallback and not self.mediapipe_available:
            fallback_bboxes = self.detect_fallback_bboxes(frame, scale)
            if fallback_bboxes and DEBUG_MODE:
                debug_log(f"Fallback detection — {len(fallback_bboxes)} face(s) on inference frame")

        worker_landmarks = self.landmarks.detect(frame)
        if not worker_landmarks:
            worker_landmarks = [
                LandmarkProvider.synthetic_landmarks(bbox)
                for bbox in self._detect_haar(frame)
            ]

        updates: List[FaceAnalysisResult] = []
        for tracked in tracked_faces:
            cached = face_cache.get(tracked.track_id)
            if cached is None:
                cached = FaceAnalysisResult(
                    id=tracked.track_id,
                    bounding_box=tracked.bounding_box,
                    timestamp=timestamp,
                )

            schedule = schedules.setdefault(tracked.track_id, TrackScheduleState())
            infer_bbox = tracked.bounding_box
            small_bbox = to_inference_bbox(infer_bbox, scale)

            landmarks = self._resolve_landmarks(small_bbox, worker_landmarks)
            result = self._clone_cached(tracked.track_id, infer_bbox, timestamp, cached)

            if frame_id - schedule.last_emotion_frame >= EMOTION_INTERVAL:
                emotion_data = self._emotion.analyze(
                    frame, small_bbox, landmarks, None, track_id=tracked.track_id,
                )
                result.emotion = str(emotion_data.get("emotion", "neutral"))
                result.confidence = float(emotion_data.get("confidence", 0.0))
                result.emotion_scores = dict(emotion_data.get("emotion_scores", {}))
                schedule.last_emotion_frame = frame_id

            aligned = align_face_crop(frame, landmarks, small_bbox) if landmarks else None
            if aligned is not None and aligned.size > 0:
                if timestamp - schedule.last_age_time >= AGE_INTERVAL_SEC:
                    age_val = self._cache.predict_age_on_crop(aligned)
                    if age_val is not None:
                        result.age = age_val
                    schedule.last_age_time = timestamp

                if timestamp - schedule.last_gender_time >= GENDER_INTERVAL_SEC:
                    gender, gender_conf = self._cache.predict_gender_on_crop(aligned)
                    if gender is not None:
                        result.gender = gender
                        result.gender_confidence = gender_conf
                    schedule.last_gender_time = timestamp

            updates.append(result)

        return updates, fallback_bboxes


def analyze_tracks_slow(
    frame: Any,
    scale: float,
    tracked_faces: List[TrackedFace],
    landmark_sets: List[LandmarkSet],
    frame_id: int,
    timestamp: float,
    schedules: Dict[int, TrackScheduleState],
    face_cache: Dict[int, FaceAnalysisResult],
    analyzer: CompositeFaceAnalyzer,
    *,
    run_fallback: bool,
) -> Tuple[List[FaceAnalysisResult], List[BoundingBox], Dict[int, TrackScheduleState], bool]:
    try:
        updates, fallback = analyzer.update_tracks_slow(
            frame, scale, tracked_faces, landmark_sets, frame_id, timestamp,
            schedules, face_cache, run_fallback=run_fallback,
        )
        return updates, fallback, dict(schedules), True
    except ImportError as exc:
        logger.warning("Missing dependency during inference: %s", exc)
    except Exception as exc:
        exc_name = type(exc).__module__
        if "tensorflow" in exc_name.lower() or "keras" in exc_name.lower():
            logger.warning("TensorFlow inference error: %s", exc)
        else:
            logger.warning("Analysis inference error: %s", exc)
    return [], [], {}, False


# =============================================================================
# Tracking
# =============================================================================

@dataclass
class _Track:
    track_id: int
    bounding_box: BoundingBox
    centroid: Tuple[float, float]
    last_seen: float
    lost_at: Optional[float] = None

    @property
    def is_visible(self) -> bool:
        return self.lost_at is None


class BlinkCounter:
    def __init__(self) -> None:
        self._counts: Dict[int, int] = {}
        self._was_closed: Dict[int, bool] = {}

    def update(self, track_id: int, eyes_closed: bool) -> int:
        was_closed = self._was_closed.get(track_id, False)
        if was_closed and not eyes_closed:
            self._counts[track_id] = self._counts.get(track_id, 0) + 1
        self._was_closed[track_id] = eyes_closed
        return self._counts.get(track_id, 0)

    def remove(self, track_id: int) -> None:
        self._counts.pop(track_id, None)
        self._was_closed.pop(track_id, None)

    def get(self, track_id: int) -> int:
        return self._counts.get(track_id, 0)


class CentroidFaceTracker:
    def __init__(
        self,
        max_distance: float = TRACK_MAX_DISTANCE,
        grace_period_sec: float = TRACK_GRACE_PERIOD_SEC,
    ) -> None:
        self.max_distance = max_distance
        self.grace_period_sec = grace_period_sec
        self._tracks: Dict[int, _Track] = {}
        self._next_id: int = 1

    def update(self, detections: List[BoundingBox], timestamp: Optional[float] = None) -> List[TrackedFace]:
        now = timestamp if timestamp is not None else time.time()
        self._purge_expired(now)

        if not detections:
            self._mark_all_lost(now)
            return self.get_visible_tracks(now)

        candidate_tracks = list(self._tracks.values())
        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        matches: List[Tuple[int, int]] = []

        distances: List[Tuple[float, int, _Track]] = []
        for det_idx, box in enumerate(detections):
            for track in candidate_tracks:
                distance = centroid_distance(box, track.bounding_box)
                if distance <= self.max_distance:
                    distances.append((distance, det_idx, track))

        distances.sort(key=lambda item: item[0])

        for distance, det_idx, track in distances:
            if det_idx in matched_detection_indices or track.track_id in matched_track_ids:
                continue
            matches.append((det_idx, track.track_id))
            matched_detection_indices.add(det_idx)
            matched_track_ids.add(track.track_id)

        for det_idx, track_id in matches:
            box = detections[det_idx]
            track = self._tracks[track_id]
            track.bounding_box = smooth_bbox(track.bounding_box, box)
            track.centroid = centroid(track.bounding_box)
            track.last_seen = now
            track.lost_at = None

        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids and track.is_visible:
                track.lost_at = now

        for det_idx, box in enumerate(detections):
            if det_idx in matched_detection_indices:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(
                track_id=track_id,
                bounding_box=box,
                centroid=centroid(box),
                last_seen=now,
            )

        return self.get_visible_tracks(now)

    def get_visible_tracks(self, timestamp: Optional[float] = None) -> List[TrackedFace]:
        now = timestamp if timestamp is not None else time.time()
        self._purge_expired(now)
        visible = [
            TrackedFace(
                track_id=track.track_id,
                bounding_box=track.bounding_box,
                centroid=track.centroid,
                last_seen=track.last_seen,
            )
            for track in self._tracks.values()
            if track.is_visible
        ]
        visible.sort(key=lambda face: face.track_id)
        return visible

    def active_track_ids(self, timestamp: Optional[float] = None) -> set[int]:
        now = timestamp if timestamp is not None else time.time()
        self._purge_expired(now)
        return set(self._tracks.keys())

    def _mark_all_lost(self, timestamp: float) -> None:
        for track in self._tracks.values():
            if track.is_visible:
                track.lost_at = timestamp

    def _purge_expired(self, timestamp: float) -> None:
        expired_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.lost_at is not None and (timestamp - track.lost_at) >= self.grace_period_sec
        ]
        for track_id in expired_ids:
            del self._tracks[track_id]


def apply_analysis_smoothing(
    smoothing: AnalysisSmoothingState,
    track_id: int,
    raw: FaceAnalysisResult,
    bounding_box: BoundingBox,
    blink_counter: BlinkCounter,
) -> FaceAnalysisResult:
    history = smoothing.histories.setdefault(track_id, deque(maxlen=EMOTION_HISTORY_SIZE))
    emotion_key = raw.emotion if raw.emotion not in {"unknown", "neutral"} else "neutral"
    if raw.confidence >= EMOTION_CONFIDENCE_THRESHOLD:
        history.append(raw.emotion)
    stable_emotion = majority_emotion(history) if history else raw.emotion
    if raw.confidence < EMOTION_CONFIDENCE_THRESHOLD and not history:
        stable_emotion = raw.emotion
    stable_confidence = raw.emotion_scores.get(stable_emotion, raw.confidence)

    scalars = smoothing.scalar_cache.setdefault(track_id, {})
    if raw.age is not None:
        prev_age = scalars.get("age")
        smoothed_age = smooth_scalar(prev_age, float(raw.age))
        if smoothed_age is not None:
            scalars["age"] = smoothed_age
    smoothed_yaw = smooth_scalar(scalars.get("yaw"), raw.yaw)
    if smoothed_yaw is not None:
        scalars["yaw"] = smoothed_yaw
    smoothed_pitch = smooth_scalar(scalars.get("pitch"), raw.pitch)
    if smoothed_pitch is not None:
        scalars["pitch"] = smoothed_pitch
    smoothed_roll = smooth_scalar(scalars.get("roll"), raw.roll)
    if smoothed_roll is not None:
        scalars["roll"] = smoothed_roll
    smoothed_smile_confidence = smooth_scalar(scalars.get("smile_confidence"), raw.smile_confidence)
    scalars["smile_confidence"] = (
        smoothed_smile_confidence if smoothed_smile_confidence is not None else scalars.get("smile_confidence", raw.smile_confidence)
    )
    smoothed_quality_score = smooth_scalar(scalars.get("quality_score"), raw.quality_score)
    scalars["quality_score"] = (
        smoothed_quality_score if smoothed_quality_score is not None else scalars.get("quality_score", raw.quality_score)
    )

    blink_count = blink_counter.get(track_id)
    age_val = int(round(scalars["age"])) if scalars.get("age") is not None else raw.age

    return FaceAnalysisResult(
        id=track_id,
        bounding_box=bounding_box,
        timestamp=raw.timestamp,
        emotion=stable_emotion,
        confidence=stable_confidence,
        emotion_scores=raw.emotion_scores,
        age=age_val,
        gender=raw.gender,
        gender_confidence=raw.gender_confidence,
        yaw=scalars.get("yaw", raw.yaw),
        pitch=scalars.get("pitch", raw.pitch),
        roll=scalars.get("roll", raw.roll),
        orientation=raw.orientation,
        left_eye=raw.left_eye,
        right_eye=raw.right_eye,
        eyes_state=raw.eyes_state,
        blink_count=blink_count,
        smiling=raw.smiling,
        smile_confidence=scalars.get("smile_confidence", raw.smile_confidence),
        quality_score=scalars.get("quality_score", raw.quality_score),
        quality_label=raw.quality_label,
        eyes_closed=raw.eyes_closed,
    )


def copy_cached_face(
    cached: FaceAnalysisResult,
    bounding_box: BoundingBox,
    blink_counter: BlinkCounter,
    track_id: int,
) -> FaceAnalysisResult:
    return FaceAnalysisResult(
        id=track_id,
        bounding_box=bounding_box,
        timestamp=cached.timestamp,
        emotion=cached.emotion,
        confidence=cached.confidence,
        emotion_scores=cached.emotion_scores,
        age=cached.age,
        gender=cached.gender,
        gender_confidence=cached.gender_confidence,
        yaw=cached.yaw,
        pitch=cached.pitch,
        roll=cached.roll,
        orientation=cached.orientation,
        left_eye=cached.left_eye,
        right_eye=cached.right_eye,
        eyes_state=cached.eyes_state,
        blink_count=blink_counter.get(track_id),
        smiling=cached.smiling,
        smile_confidence=cached.smile_confidence,
        quality_score=cached.quality_score,
        quality_label=cached.quality_label,
        eyes_closed=cached.eyes_closed,
    )


def merge_face_update(existing: FaceAnalysisResult, update: FaceAnalysisResult) -> FaceAnalysisResult:
    has_emotion_update = bool(update.emotion_scores) or update.confidence > 0
    return FaceAnalysisResult(
        id=existing.id,
        bounding_box=existing.bounding_box,
        timestamp=update.timestamp,
        emotion=update.emotion if has_emotion_update else existing.emotion,
        confidence=update.confidence if has_emotion_update else existing.confidence,
        emotion_scores=update.emotion_scores if update.emotion_scores else existing.emotion_scores,
        age=update.age if update.age is not None else existing.age,
        gender=update.gender if update.gender is not None else existing.gender,
        gender_confidence=update.gender_confidence if update.gender_confidence is not None else existing.gender_confidence,
        yaw=update.yaw if update.yaw is not None else existing.yaw,
        pitch=update.pitch if update.pitch is not None else existing.pitch,
        roll=update.roll if update.roll is not None else existing.roll,
        orientation=update.orientation if update.orientation != _ESTIMATING else existing.orientation,
        left_eye=update.left_eye if update.left_eye != _ESTIMATING else existing.left_eye,
        right_eye=update.right_eye if update.right_eye != _ESTIMATING else existing.right_eye,
        eyes_state=update.eyes_state if update.eyes_state != _ESTIMATING else existing.eyes_state,
        blink_count=existing.blink_count,
        smiling=update.smile_confidence > 0 or update.smiling or existing.smiling,
        smile_confidence=update.smile_confidence if update.smile_confidence > 0 else existing.smile_confidence,
        quality_score=update.quality_score if update.quality_score > 0 else existing.quality_score,
        quality_label=update.quality_label or existing.quality_label,
        eyes_closed=update.eyes_closed,
    )


class RenderSmoother:
    """Smooths bounding boxes and confidence bars for flicker-free UI."""

    def __init__(self) -> None:
        self._bboxes: Dict[int, BoundingBox] = {}
        self._confidence: Dict[int, float] = {}
        self._smile_confidence: Dict[int, float] = {}

    def smooth(self, face: FaceAnalysisResult) -> FaceAnalysisResult:
        track_id = face.id
        self._bboxes[track_id] = smooth_bbox(self._bboxes.get(track_id), face.bounding_box, UI_BBOX_SMOOTH_ALPHA)
        prev_conf = self._confidence.get(track_id)
        self._confidence[track_id] = smooth_scalar(prev_conf, face.confidence, UI_CONF_SMOOTH_ALPHA) or face.confidence
        prev_smile = self._smile_confidence.get(track_id)
        self._smile_confidence[track_id] = (
            smooth_scalar(prev_smile, face.smile_confidence, UI_CONF_SMOOTH_ALPHA) or face.smile_confidence
        )

        return FaceAnalysisResult(
            id=face.id,
            bounding_box=self._bboxes[track_id],
            timestamp=face.timestamp,
            emotion=face.emotion,
            confidence=self._confidence[track_id],
            emotion_scores=face.emotion_scores,
            age=face.age,
            gender=face.gender,
            gender_confidence=face.gender_confidence,
            yaw=face.yaw,
            pitch=face.pitch,
            roll=face.roll,
            orientation=face.orientation,
            left_eye=face.left_eye,
            right_eye=face.right_eye,
            eyes_state=face.eyes_state,
            blink_count=face.blink_count,
            smiling=face.smiling,
            smile_confidence=self._smile_confidence[track_id],
            quality_score=face.quality_score,
            quality_label=face.quality_label,
            eyes_closed=face.eyes_closed,
        )

    def remove(self, track_id: int) -> None:
        self._bboxes.pop(track_id, None)
        self._confidence.pop(track_id, None)
        self._smile_confidence.pop(track_id, None)


def purge_expired_face_cache(
    tracker: CentroidFaceTracker,
    face_cache: Dict[int, FaceAnalysisResult],
    smoothing: AnalysisSmoothingState,
    blink_counter: BlinkCounter,
    render_smoother: RenderSmoother,
    timestamp: float,
) -> None:
    valid_ids = tracker.active_track_ids(timestamp)
    for track_id in list(face_cache.keys()):
        if track_id not in valid_ids:
            del face_cache[track_id]
            smoothing.histories.pop(track_id, None)
            smoothing.scalar_cache.pop(track_id, None)
            blink_counter.remove(track_id)
            render_smoother.remove(track_id)


# =============================================================================
# Live Analytics
# =============================================================================

class _SmoothScalar:
    __slots__ = ("_alpha", "_value", "_initialized")

    def __init__(self, alpha: float = 0.22) -> None:
        self._alpha = alpha
        self._value = 0.0
        self._initialized = False

    def update(self, sample: float) -> float:
        if not self._initialized:
            self._value = sample
            self._initialized = True
        else:
            self._value = self._alpha * sample + (1.0 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value

    @property
    def initialized(self) -> bool:
        return self._initialized


class _SmoothDistribution:
    def __init__(self, alpha: float = 0.18) -> None:
        self._alpha = alpha
        self._values: Dict[str, float] = {key: 0.0 for key in EMOTION_KEYS}

    def update(self, sample: Dict[str, float]) -> Dict[str, float]:
        for key in EMOTION_KEYS:
            target = sample.get(key, 0.0)
            current = self._values[key]
            if current == 0.0 and target > 0.0:
                self._values[key] = target
            else:
                self._values[key] = self._alpha * target + (1.0 - self._alpha) * current
        return dict(self._values)


class LiveAnalytics:
    def __init__(self) -> None:
        self._fps = _SmoothScalar(0.25)
        self._avg_fps = _SmoothScalar(0.12)
        self._inference_ms = _SmoothScalar(0.15)
        self._render_ms = _SmoothScalar(0.15)
        self._cpu = _SmoothScalar(0.08)
        self._distribution = _SmoothDistribution(0.18)
        self._cpu_available = False
        self._last_cpu_sample = 0.0
        self._cpu_interval_sec = 1.0
        self._memory_interval_sec = 1.0
        self._last_memory_sample = 0.0
        self._memory = _SmoothScalar(0.08)
        self._model_ready = False
        self._snapshot = LiveDashboardSnapshot(
            emotion_distribution={key: 0.0 for key in EMOTION_KEYS},
            emotion_counts={key: 0 for key in EMOTION_KEYS},
            target_fps=TARGET_FPS,
        )

        if _psutil_mod is not None:
            _psutil_mod.cpu_percent(interval=None)
            self._cpu_available = True
        else:
            self._cpu_available = False

    @property
    def snapshot(self) -> LiveDashboardSnapshot:
        return self._snapshot

    def mark_model_ready(self) -> None:
        self._model_ready = True

    @staticmethod
    def _aggregate_distribution(faces: List[FaceAnalysisResult]) -> Tuple[Dict[str, float], Dict[str, int]]:
        if not faces:
            return {key: 0.0 for key in EMOTION_KEYS}, {key: 0 for key in EMOTION_KEYS}

        totals = {key: 0.0 for key in EMOTION_KEYS}
        counts = {key: 0 for key in EMOTION_KEYS}
        for face in faces:
            emotion = face.emotion.lower()
            if emotion in counts:
                counts[emotion] += 1
            scores = face.emotion_scores or {}
            for key in EMOTION_KEYS:
                totals[key] += float(scores.get(key, 0.0))

        count = len(faces)
        distribution = {key: totals[key] / count for key in EMOTION_KEYS}
        return distribution, counts

    def update(
        self,
        *,
        fps: float,
        avg_fps: float,
        inference_ms: float,
        render_ms: float,
        faces: List[FaceAnalysisResult],
        current_faces: int,
        total_tracked: int,
        camera_connected: bool,
        timestamp: float,
    ) -> LiveDashboardSnapshot:
        raw_distribution, raw_counts = self._aggregate_distribution(faces)
        if current_faces == 0:
            raw_distribution = {
                key: self._snapshot.emotion_distribution.get(key, 0.0) * 0.92 for key in EMOTION_KEYS
            }
            raw_counts = {key: 0 for key in EMOTION_KEYS}

        smoothed_distribution = self._distribution.update(raw_distribution)

        cpu_display: Optional[float] = None
        memory_display: Optional[float] = None
        if self._cpu_available and timestamp - self._last_cpu_sample >= self._cpu_interval_sec:
            sample = read_cpu_percent()
            self._last_cpu_sample = timestamp
            if sample is not None:
                cpu_display = self._cpu.update(sample)
        elif self._cpu_available and self._cpu.initialized:
            cpu_display = self._cpu.value

        if self._cpu_available and timestamp - self._last_memory_sample >= self._memory_interval_sec:
            mem_sample = read_memory_percent()
            self._last_memory_sample = timestamp
            if mem_sample is not None:
                memory_display = self._memory.update(mem_sample)
        elif self._cpu_available and self._memory.initialized:
            memory_display = self._memory.value

        self._snapshot = LiveDashboardSnapshot(
            fps=self._fps.update(fps),
            avg_fps=self._avg_fps.update(avg_fps),
            current_faces=current_faces,
            total_tracked=total_tracked,
            inference_ms=self._inference_ms.update(inference_ms),
            render_ms=self._render_ms.update(render_ms),
            cpu_percent=cpu_display,
            memory_percent=memory_display,
            target_fps=TARGET_FPS,
            emotion_distribution=smoothed_distribution,
            emotion_counts=raw_counts,
            camera_connected=camera_connected,
            ai_ready=self._model_ready,
            tracking_running=total_tracked > 0,
        )
        return self._snapshot


# =============================================================================
# Rendering & Dashboard UI
# =============================================================================

def emotion_color(emotion: str) -> Tuple[int, int, int]:
    return EMOTION_COLORS.get(emotion.lower(), (180, 180, 180))


def rounded_rect_mask(height: int, width: int, radius: int) -> Any:
    radius = max(1, min(radius, height // 2, width // 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (radius, radius), radius, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1, lineType=cv2.LINE_AA)
    return mask


def blend_panel(
    frame: Any,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    alpha: float,
    radius: int,
) -> None:
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return

    roi = frame[y1:y2, x1:x2]
    local_h, local_w = roi.shape[:2]
    mask = rounded_rect_mask(local_h, local_w, radius)
    mask_f = (mask.astype(np.float32) / 255.0) * alpha
    mask_3 = cv2.merge([mask_f, mask_f, mask_f])
    panel = np.full_like(roi, color, dtype=np.uint8)
    blended = (panel * mask_3 + roi.astype(np.float32) * (1.0 - mask_3)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended


def draw_rounded_border(
    frame: Any,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int,
    radius: int,
) -> None:
    if x2 <= x1 or y2 <= y1:
        return
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius <= 1:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        return

    t = thickness
    cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, t, cv2.LINE_AA)
    cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, t, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, t, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, t, cv2.LINE_AA)


def text_size(text: str, scale: float, thickness: int) -> Tuple[int, int]:
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return width, height + baseline


def put_text_aa(
    frame: Any,
    text: str,
    origin: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    x, y = origin
    if shadow:
        cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_confidence_bar(
    frame: Any,
    x: int, y: int,
    width: int, height: int,
    pct: float,
    color: Tuple[int, int, int],
) -> None:
    pct = max(0.0, min(100.0, pct))
    cv2.rectangle(frame, (x, y), (x + width, y + height), BAR_TRACK, -1, lineType=cv2.LINE_AA)
    fill_w = max(1, int(width * pct / 100.0))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color, -1, lineType=cv2.LINE_AA)


def draw_status_row(
    frame: Any,
    x: int, y: int,
    label: str,
    active: bool,
    scale: float,
) -> None:
    dot_color = STATUS_OK if active else STATUS_OFF
    cv2.circle(frame, (x + 6, y - 5), 5, dot_color, -1, lineType=cv2.LINE_AA)
    put_text_aa(frame, label, (x + 18, y), scale, TEXT_PRIMARY if active else TEXT_MUTED, 1, shadow=False)


class DashboardRenderer:
    def render(
        self,
        frame: Any,
        faces: Sequence[FaceRenderable],
        dashboard: LiveDashboardSnapshot,
    ) -> None:
        for face in faces:
            self._draw_face_card(frame, face)
        self._draw_top_left_metrics(frame, dashboard)
        self._draw_emotion_distribution(frame, dashboard)
        self._draw_bottom_left_shortcuts(frame)
        self._draw_bottom_right_status(frame, dashboard)

    def _draw_face_card(self, frame: Any, face: FaceRenderable) -> None:
        x, y, w, h = face.bounding_box
        if w <= 0 or h <= 0:
            return
        color = emotion_color(face.emotion)
        emotion_key = face.emotion.lower()
        emoji = EMOTION_EMOJI.get(emotion_key, EMOTION_EMOJI.get("neutral", ""))
        name = display_emotion(face.emotion, face.confidence, face.emotion_scores)
        confidence_text = f"{face.confidence:.0f}%" if face.confidence > 0 else _ESTIMATING
        id_text = f"Face #{face.id}"

        height, width = frame.shape[:2]

        bx1, by1 = max(0, x), max(0, y)
        bx2, by2 = min(width - 1, x + w), min(height - 1, y + h)
        draw_rounded_border(frame, bx1, by1, bx2, by2, color, FACE_BORDER_THICKNESS, FACE_BORDER_RADIUS)

        gender_conf = (
            f" ({face.gender_confidence:.0f}%)"
            if face.gender_confidence is not None
            else ""
        )
        gender_text = display_label(face.gender)
        age_text = display_age(face.age)
        smile_text = "Yes" if face.smiling else "No"
        pose_text = _ESTIMATING
        if face.yaw is not None and face.pitch is not None and face.roll is not None:
            pose_text = f"Yaw {face.yaw:.0f}  Pitch {face.pitch:.0f}  Roll {face.roll:.0f}"

        detail_lines = [
            f"Emotion: {name} ({confidence_text})",
            f"Age: {age_text}",
            f"Gender: {gender_text}{gender_conf}",
            f"Orientation: {face.orientation.replace('Looking ', '')}",
            f"Head Pose: {pose_text}",
            f"Smile: {smile_text} ({face.smile_confidence:.0f}%)",
            f"Eyes: {face.eyes_state} (L:{face.left_eye} R:{face.right_eye})",
            f"Blinks: {face.blink_count}",
        ]

        pad_x, pad_y = 12, 10
        line_gap = 4
        id_scale, detail_scale = 0.42, 0.38
        id_th, detail_th = 1, 1

        id_w, id_h = text_size(id_text, id_scale, id_th)
        detail_w = max(text_size(line, detail_scale, detail_th)[0] for line in detail_lines)
        card_w = max(id_w, detail_w) + pad_x * 2
        card_h = pad_y + id_h + line_gap + len(detail_lines) * 16 + pad_y + 12

        card_x1 = x + w + 10
        if card_x1 + card_w > width - 8:
            card_x1 = max(0, min(x, width - card_w))
            card_y2 = max(0, y - 8)
            card_y1 = max(0, card_y2 - card_h)
        else:
            card_y1 = max(8, min(y, height - card_h - 8))
        card_x1 = max(8, min(card_x1, width - card_w - 8))
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h

        blend_panel(frame, card_x1, card_y1, card_x2, card_y2, PANEL_BG, PANEL_ALPHA, CORNER_RADIUS)
        draw_rounded_border(frame, card_x1, card_y1, card_x2, card_y2, color, 2, CORNER_RADIUS)

        accent_x2 = card_x2 - CORNER_RADIUS
        accent_x1 = card_x1 + CORNER_RADIUS
        if accent_x2 > accent_x1:
            cv2.rectangle(frame, (accent_x1, card_y1), (accent_x2, card_y1 + 4), color, -1, lineType=cv2.LINE_AA)

        cursor_x = card_x1 + pad_x
        cursor_y = card_y1 + pad_y + id_h
        put_text_aa(frame, id_text, (cursor_x, cursor_y), id_scale, TEXT_MUTED, id_th)

        header = f"{emoji} {name}".strip() if name != _ESTIMATING else f"{emoji} {_ESTIMATING}".strip()
        cursor_y += line_gap + 14
        put_text_aa(frame, header, (cursor_x, cursor_y), 0.48, color, 1)

        bar_y = cursor_y + 6
        bar_w = card_w - pad_x * 2
        conf_value = face.confidence if face.confidence > 0 else 0.0
        draw_confidence_bar(frame, cursor_x, bar_y, bar_w, 6, conf_value, color)

        for line in detail_lines:
            cursor_y += 16
            put_text_aa(frame, line, (cursor_x, cursor_y), detail_scale, TEXT_SECONDARY, detail_th, shadow=False)

    def _draw_top_left_metrics(self, frame: Any, dash: LiveDashboardSnapshot) -> None:
        pad = 14
        line_h = 22
        scale = 0.46
        thickness = 1
        cpu_text = f"{dash.cpu_percent:.0f}%" if dash.cpu_percent is not None else "Unavailable"
        memory_text = f"{dash.memory_percent:.0f}%" if dash.memory_percent is not None else "Unavailable"

        rows = [
            ("FPS", f"{dash.fps:.0f}"),
            ("Avg FPS", f"{dash.avg_fps:.0f}"),
            ("Target FPS", f"{dash.target_fps:.0f}"),
            ("Faces", f"{dash.current_faces}"),
            ("Inference", f"{dash.inference_ms:.0f} ms"),
            ("Render", f"{dash.render_ms:.0f} ms"),
            ("CPU", cpu_text),
            ("Memory", memory_text),
        ]

        label_w = max(text_size(label, scale, thickness)[0] for label, _ in rows)
        value_w = max(text_size(value, scale, thickness)[0] for _, value in rows)
        panel_w = pad * 2 + label_w + 16 + value_w
        panel_h = pad * 2 + line_h * len(rows) - 2

        x1, y1 = 16, 16
        x2, y2 = x1 + panel_w, y1 + panel_h

        blend_panel(frame, x1, y1, x2, y2, PANEL_BG_ALT, PANEL_ALPHA, CORNER_RADIUS)
        draw_rounded_border(frame, x1, y1, x2, y2, (70, 70, 80), 1, CORNER_RADIUS)
        put_text_aa(frame, "LIVE METRICS", (x1 + pad, y1 + 16), 0.40, TEXT_MUTED, 1, shadow=False)

        for index, (label, value) in enumerate(rows):
            y = y1 + pad + 10 + (index + 1) * line_h
            put_text_aa(frame, label, (x1 + pad, y), scale, TEXT_SECONDARY, thickness, shadow=False)
            put_text_aa(frame, value, (x1 + pad + label_w + 16, y), scale, TEXT_PRIMARY, thickness, shadow=False)

    def _draw_emotion_distribution(self, frame: Any, dash: LiveDashboardSnapshot) -> None:
        height, width = frame.shape[:2]
        pad = 14
        title_scale = 0.42
        label_scale = 0.38
        pct_scale = 0.36
        row_h = 22
        bar_h = 8
        bar_max_w = 110

        panel_w = 250
        panel_h = pad * 2 + 20 + row_h * len(EMOTION_KEYS)

        x2 = width - 16
        x1 = x2 - panel_w
        y1 = 16
        y2 = y1 + panel_h

        blend_panel(frame, x1, y1, x2, y2, PANEL_BG_ALT, PANEL_ALPHA, CORNER_RADIUS)
        draw_rounded_border(frame, x1, y1, x2, y2, (70, 70, 80), 1, CORNER_RADIUS)
        put_text_aa(
            frame, "EMOTION DISTRIBUTION",
            (x1 + pad, y1 + 16), title_scale, TEXT_MUTED, 1, shadow=False,
        )

        for index, key in enumerate(EMOTION_KEYS):
            pct = dash.emotion_distribution.get(key, 0.0)
            count = dash.emotion_counts.get(key, 0)
            color = emotion_color(key)
            row_y = y1 + pad + 22 + index * row_h + 14

            label = f"{key.title()} ({count})"
            pct_text = f"{pct:.0f}%"
            put_text_aa(frame, label, (x1 + pad, row_y), label_scale, TEXT_SECONDARY, 1, shadow=False)

            bar_x = x1 + pad + 92
            draw_confidence_bar(frame, bar_x, row_y - 10, bar_max_w, bar_h, pct, color)
            put_text_aa(frame, pct_text, (bar_x + bar_max_w + 8, row_y), pct_scale, TEXT_PRIMARY, 1, shadow=False)

    def _draw_bottom_left_shortcuts(self, frame: Any) -> None:
        height = frame.shape[:2][0]
        pad_x, pad_y = 14, 12
        scale = 0.44
        line_h = 20

        lines = ["Press Q to Quit"]
        panel_w = 170
        panel_h = pad_y * 2 + line_h + 4

        x1, y2 = 16, height - 16
        y1 = y2 - panel_h
        x2 = x1 + panel_w

        blend_panel(frame, x1, y1, x2, y2, PANEL_BG_ALT, PANEL_ALPHA * 0.88, CORNER_RADIUS)
        draw_rounded_border(frame, x1, y1, x2, y2, (70, 70, 80), 1, CORNER_RADIUS)

        y = y1 + pad_y + line_h + 2
        put_text_aa(frame, lines[0], (x1 + pad_x, y), scale, TEXT_SECONDARY, 1, shadow=False)

    def _draw_bottom_right_status(self, frame: Any, dash: LiveDashboardSnapshot) -> None:
        height, width = frame.shape[:2]
        pad_x, pad_y = 14, 12
        scale = 0.44
        line_h = 22

        rows = [
            ("Camera Connected", dash.camera_connected),
            ("AI Ready", dash.ai_ready),
            ("Tracking Running", dash.tracking_running),
        ]

        panel_w = 220
        panel_h = pad_y * 2 + 18 + line_h * len(rows)

        x2, y2 = width - 16, height - 16
        x1 = x2 - panel_w
        y1 = y2 - panel_h

        blend_panel(frame, x1, y1, x2, y2, PANEL_BG_ALT, PANEL_ALPHA * 0.88, CORNER_RADIUS)
        draw_rounded_border(frame, x1, y1, x2, y2, (70, 70, 80), 1, CORNER_RADIUS)
        put_text_aa(frame, "STATUS", (x1 + pad_x, y1 + 16), 0.40, TEXT_MUTED, 1, shadow=False)

        for index, (label, active) in enumerate(rows):
            y = y1 + pad_y + 22 + (index + 1) * line_h
            draw_status_row(frame, x1 + pad_x, y, label, active, scale)


_dashboard_renderer = DashboardRenderer()


def render_dashboard(
    frame: Any,
    faces: Sequence[FaceRenderable],
    dashboard: LiveDashboardSnapshot,
) -> None:
    _dashboard_renderer.render(frame, faces, dashboard)


# =============================================================================
# Terminal Dashboard (optional refresh)
# =============================================================================

def print_dashboard(state: AppState, perf: PerformanceTracker, analytics: LiveAnalytics) -> None:
    now = time.time()
    if now - state.last_dashboard_time < DASHBOARD_REFRESH_SEC:
        return
    state.last_dashboard_time = now

    dash = analytics.snapshot
    cpu_line = f"{dash.cpu_percent:.0f}%" if dash.cpu_percent is not None else "Unavailable"
    mem_line = f"{dash.memory_percent:.0f}%" if dash.memory_percent is not None else "Unavailable"

    lines = [
        "=" * 54,
        f"FPS: {dash.fps:.0f}  |  Avg: {dash.avg_fps:.0f}  |  Target: {dash.target_fps:.0f}",
        f"Inference: {dash.inference_ms:.0f} ms  |  Render: {dash.render_ms:.0f} ms",
        f"CPU: {cpu_line}  |  Memory: {mem_line}",
        f"Faces: {dash.current_faces}  |  Tracked: {dash.total_tracked}",
        "",
        "EMOTION DISTRIBUTION",
    ]

    for key in EMOTION_KEYS:
        pct = dash.emotion_distribution.get(key, 0.0)
        count = dash.emotion_counts.get(key, 0)
        bar = "#" * int(pct / 5)
        lines.append(f"  {key.title():<10} {count:2d}  {pct:5.1f}%  {bar}")

    lines.extend(["", "=" * 54])
    output = "\n".join(lines)

    if sys.stdout.isatty() and state.dashboard_terminal_lines > 0:
        sys.stdout.write(f"\033[{state.dashboard_terminal_lines}A\033[J")
    print(output, flush=True)
    state.dashboard_terminal_lines = len(lines)


# =============================================================================
# Main Loop
# =============================================================================

def apply_inference_result(
    inf: InferencePacket,
    face_cache: Dict[int, FaceAnalysisResult],
    schedules: Dict[int, TrackScheduleState],
    smoothing: AnalysisSmoothingState,
    blink_counter: BlinkCounter,
    analytics: LiveAnalytics,
) -> None:
    if inf.success:
        analytics.mark_model_ready()

    for track_id, schedule in inf.schedule_updates.items():
        local = schedules.setdefault(track_id, TrackScheduleState())
        local.last_emotion_frame = schedule.last_emotion_frame
        local.last_age_time = schedule.last_age_time
        local.last_gender_time = schedule.last_gender_time

    for update in inf.face_updates:
        existing = face_cache.get(update.id)
        merged = merge_face_update(existing, update) if existing is not None else update
        face_cache[update.id] = apply_analysis_smoothing(
            smoothing, update.id, merged, merged.bounding_box, blink_counter,
        )
        if DEBUG_MODE and update.emotion_scores:
            debug_log(
                f"UI update — Face ID {update.id}: {merged.emotion} "
                f"({merged.confidence:.1f}%)"
            )


def build_display_faces(
    tracked_faces: List[TrackedFace],
    face_cache: Dict[int, FaceAnalysisResult],
    blink_counter: BlinkCounter,
    render_smoother: RenderSmoother,
) -> List[FaceAnalysisResult]:
    faces: List[FaceAnalysisResult] = []
    for tracked in tracked_faces:
        cached = face_cache.get(tracked.track_id)
        if cached is None:
            placeholder = FaceAnalysisResult(
                id=tracked.track_id,
                bounding_box=tracked.bounding_box,
                timestamp=time.time(),
            )
            faces.append(render_smoother.smooth(placeholder))
            continue

        face = copy_cached_face(cached, tracked.bounding_box, blink_counter, tracked.track_id)
        faces.append(render_smoother.smooth(face))

    faces.sort(key=lambda item: item.id)
    return faces


def run() -> None:
    startup_log("INFO", f"Starting {APP_NAME}...")
    startup_log("INFO", f"Welcome back, {USER_NAME}.")

    if not check_dependencies():
        sys.exit(1)

    model_cache = ModelCache()
    if not model_cache.prepare_and_load():
        if model_cache.user_declined_download:
            startup_log("INFO", "Startup cancelled.")
        else:
            startup_log(
                "ERROR",
                f"{USER_NAME}, the application cannot continue until the required resource is available.",
            )
        sys.exit(1)

    startup_log("INFO", "Initializing camera...")
    cap = initialize_camera()
    if cap is None:
        cleanup(None)
        sys.exit(1)

    analyzer = CompositeFaceAnalyzer(model_cache, detector_backend=DETECTOR_BACKEND)
    tracker = CentroidFaceTracker(max_distance=TRACK_MAX_DISTANCE, grace_period_sec=TRACK_GRACE_PERIOD_SEC)
    smoothing = AnalysisSmoothingState()
    blink_counter = BlinkCounter()
    state = AppState()
    face_cache: Dict[int, FaceAnalysisResult] = {}
    perf = PerformanceTracker()
    display_buffer = DisplayBuffer()
    analytics = LiveAnalytics()
    analytics.mark_model_ready()

    frame_buffer = LatestFrameBuffer()
    inference_results = LatestInferenceResult()

    capture_thread = CameraCaptureThread(cap, frame_buffer, max_failures=MAX_READ_FAILURES)
    inference_worker = InferenceWorker(analyzer=analyzer, results=inference_results)

    capture_thread.start()
    inference_worker.start()
    logger.info("Pipeline started (capture thread + inference worker).")

    schedules: Dict[int, TrackScheduleState] = {}
    render_smoother = RenderSmoother()
    fallback_bboxes: List[BoundingBox] = []
    latest_inference_ms = 0.0
    last_applied_inference_id = -1
    last_frame_packet: Optional[FramePacket] = None

    try:
        while True:
            frame_start = time.perf_counter()

            if not capture_thread.is_alive() and capture_thread.failures >= MAX_READ_FAILURES:
                logger.error("Capture thread stopped — camera unavailable.")
                break

            packet = frame_buffer.consume()
            if packet is not None:
                last_frame_packet = packet

            if last_frame_packet is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested by user.")
                    break
                elapsed = time.perf_counter() - frame_start
                sleep_time = TARGET_FRAME_TIME_SEC - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            packet = last_frame_packet
            now = packet.timestamp
            frame_id = packet.frame_id

            try:
                detect_bboxes, landmark_sets = analyzer.detect_faces(packet.frame)
            except cv2.error as exc:
                logger.debug("Face detection error: %s", exc)
                detect_bboxes, landmark_sets = [], []

            if detect_bboxes:
                bboxes = [
                    sanitized
                    for bbox in detect_bboxes
                    if (sanitized := sanitize_bbox(bbox, packet.frame.shape)) is not None
                ]
            elif fallback_bboxes:
                bboxes = [
                    sanitized
                    for bbox in fallback_bboxes
                    if (sanitized := sanitize_bbox(bbox, packet.frame.shape)) is not None
                ]
            else:
                bboxes = []

            tracked_faces = tracker.update(bboxes, timestamp=now)
            if DEBUG_MODE and tracked_faces:
                debug_log(f"Tracking — {len(tracked_faces)} face(s) active")

            inf = inference_results.latest()
            if inf is not None and inf.frame_id > last_applied_inference_id:
                latest_inference_ms = inf.inference_ms
                apply_inference_result(
                    inf, face_cache, schedules, smoothing, blink_counter, analytics,
                )
                if inf.fallback_bboxes:
                    fallback_bboxes = inf.fallback_bboxes
                last_applied_inference_id = inf.frame_id

            for tracked in tracked_faces:
                landmarks = CompositeFaceAnalyzer._resolve_landmarks(tracked.bounding_box, landmark_sets)
                cached = face_cache.get(
                    tracked.track_id,
                    FaceAnalysisResult(id=tracked.track_id, bounding_box=tracked.bounding_box, timestamp=now),
                )
                schedule = schedules.setdefault(tracked.track_id, TrackScheduleState())
                updated = analyzer.update_track_fast(
                    packet.frame, tracked.track_id, tracked.bounding_box,
                    landmarks, frame_id, now, schedule, cached,
                )
                updated.blink_count = blink_counter.update(tracked.track_id, updated.eyes_closed)
                face_cache[tracked.track_id] = updated

            run_fallback = not analyzer.mediapipe_available and frame_id % FALLBACK_DETECT_INTERVAL == 0
            inference_worker.submit(
                packet.frame, frame_id, now, tracked_faces,
                landmark_sets, run_fallback, face_cache, schedules,
            )

            purge_expired_face_cache(
                tracker, face_cache, smoothing, blink_counter, render_smoother, now,
            )
            state.faces = build_display_faces(tracked_faces, face_cache, blink_counter, render_smoother)
            total_tracked = len(tracker.active_track_ids(now))

            render_start = time.perf_counter()
            display = display_buffer.prepare(packet.frame)
            dashboard = analytics.update(
                fps=perf.metrics.fps,
                avg_fps=perf.metrics.avg_fps,
                inference_ms=latest_inference_ms,
                render_ms=perf.metrics.render_ms,
                faces=state.faces,
                current_faces=len(state.faces),
                total_tracked=total_tracked,
                camera_connected=capture_thread.is_alive(),
                timestamp=now,
            )
            try:
                render_dashboard(display, state.faces, dashboard)
            except cv2.error as exc:
                logger.debug("Dashboard render error: %s", exc)
            render_ms = (time.perf_counter() - render_start) * 1000.0

            perf.tick(packet.capture_ms, render_ms, latest_inference_ms)
            print_dashboard(state, perf, analytics)

            try:
                cv2.imshow(WINDOW_NAME, display)
            except cv2.error as exc:
                logger.debug("Display error: %s", exc)
                continue
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested by user.")
                break

            elapsed = time.perf_counter() - frame_start
            sleep_time = TARGET_FRAME_TIME_SEC - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=DEBUG)
    finally:
        capture_thread.stop()
        inference_worker.stop()
        capture_thread.join(timeout=1.0)
        inference_worker.join(timeout=2.0)
        cleanup(cap)


if __name__ == "__main__":
    run()

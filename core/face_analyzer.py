# =============================================================================
# File: core/face_analyzer.py
# Mục đích: Phát hiện khuôn mặt và trích xuất đặc trưng nhận diện.
#
# Thư viện: InsightFace (gói model "buffalo_l")
#   + RetinaFace: Phát hiện khuôn mặt (bounding box + 5 điểm mốc)
#   + ArcFace: Nhận diện khuôn mặt (vector embedding 512 chiều)
#   + Landmark 2D 106: 106 điểm mốc chi tiết trên mặt
# =============================================================================

import threading
from typing import Any, Optional
import numpy as np
import cv2
import insightface
from core.model_loader import build_provider_fallback_chain, load_with_provider_fallback

_FACE_ANALYSER: Optional[Any] = None
_ANALYSER_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    """Khởi tạo bộ phân tích mặt (Singleton + Double-Check Locking)."""
    global _FACE_ANALYSER
    if _FACE_ANALYSER is None:
        with _ANALYSER_LOCK:
            if _FACE_ANALYSER is None:
                def _loader(providers):
                    analyser = insightface.app.FaceAnalysis(
                        name="buffalo_l",
                        providers=providers,
                        allowed_modules=["detection", "recognition", "landmark_2d_106"],
                    )
                    # Giảm kích thước đầu vào để tăng mạnh tốc độ nhận diện
                    analyser.prepare(ctx_id=0, det_size=(320, 320))
                    return analyser

                _FACE_ANALYSER = load_with_provider_fallback(
                    _loader,
                    build_provider_fallback_chain(),
                    "FaceAnalysis(buffalo_l)",
                )
    return _FACE_ANALYSER


def get_one_face(frame: np.ndarray) -> Optional[Any]:
    """Trả về 1 khuôn mặt chính (nằm bên trái nhất) trong khung hình."""
    faces = get_face_analyser().get(frame)
    if not faces:
        return None
    return min(faces, key=lambda f: f.bbox[0])


def get_many_faces(frame: np.ndarray) -> list:
    """Trả về TẤT CẢ khuôn mặt trong khung hình."""
    try:
        faces = get_face_analyser().get(frame)
        return faces if faces else []
    except Exception:
        return []


def get_face_embedding(face: Any) -> Optional[np.ndarray]:
    """Trích xuất vector đặc trưng 512 chiều (Face Embedding) từ khuôn mặt."""
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        return face.normed_embedding
    return None


def _create_opencv_tracker():
    """Create the best available OpenCV tracker instance.

    Notes:
        - Many Tracker* APIs live in opencv-contrib-python.
        - Newer OpenCV versions may expose trackers under cv2.legacy.*
        - If no tracker implementation exists, return None and the app will
          fall back to face detection each frame.
    """

    # Prefer MOSSE for max speed, then KCF, and CSRT as fallback.
    candidates = [
        (cv2, "TrackerMOSSE_create"),
        (getattr(cv2, "legacy", None), "TrackerMOSSE_create"),
        (cv2, "TrackerKCF_create"),
        (getattr(cv2, "legacy", None), "TrackerKCF_create"),
        (cv2, "TrackerCSRT_create"),
        (getattr(cv2, "legacy", None), "TrackerCSRT_create"),
    ]

    for module, func_name in candidates:
        if module is None:
            continue
        factory = getattr(module, func_name, None)
        if callable(factory):
            try:
                return factory()
            except Exception:
                continue

    return None

class FaceTracker:
    def __init__(self, detect_interval=15):
        self.tracker = None
        self.last_face = None
        self.frames_since_detect = 0
        self.detect_interval = detect_interval

    def update(self, frame: np.ndarray, analyzer_func) -> Optional[Any]:
        # Nếu OpenCV không có module tracking (không cài opencv-contrib),
        # fallback: detect mỗi frame để tránh crash.
        if self.tracker is None and self.last_face is None:
            # Chưa có trạng thái tracking; detect trực tiếp.
            return self._detect(frame, analyzer_func)

        # Nếu chưa có tracker, mất dấu hoặc đã đến lúc detect lại
        if self.tracker is None or self.frames_since_detect >= self.detect_interval:
            return self._detect(frame, analyzer_func)

        # Tiếp tục tracking
        success, bbox = self.tracker.update(frame)
        if not success:
            return self._detect(frame, analyzer_func)

        # Cập nhật bounding box và landmarks dựa trên độ lệch của tracker
        x, y, w, h = bbox
        old_bbox = self.last_face.bbox
        old_w = old_bbox[2] - old_bbox[0]
        old_h = old_bbox[3] - old_bbox[1]
        old_x, old_y = old_bbox[0], old_bbox[1]

        sx = w / old_w if old_w > 0 else 1.0
        sy = h / old_h if old_h > 0 else 1.0

        new_kps = self.last_face.kps.copy()
        new_kps[:, 0] = (new_kps[:, 0] - old_x) * sx + x
        new_kps[:, 1] = (new_kps[:, 1] - old_y) * sy + y

        new_landmark_106 = None
        landmark_106 = getattr(self.last_face, "landmark_2d_106", None)
        if landmark_106 is not None:
            new_landmark_106 = landmark_106.copy()
            new_landmark_106[:, 0] = (new_landmark_106[:, 0] - old_x) * sx + x
            new_landmark_106[:, 1] = (new_landmark_106[:, 1] - old_y) * sy + y

        self.last_face.bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
        self.last_face.kps = new_kps
        if new_landmark_106 is not None:
            self.last_face.landmark_2d_106 = new_landmark_106
        self.frames_since_detect += 1
        return self.last_face

    def _detect(self, frame: np.ndarray, analyzer_func) -> Optional[Any]:
        face = analyzer_func(frame)
        if face is not None:
            tracker = _create_opencv_tracker()
            if tracker is None:
                # Không có tracker (thiếu opencv-contrib) => chỉ dùng detect.
                self.tracker = None
                self.last_face = face
                self.frames_since_detect = 0
                return face

            self.tracker = tracker
            bbox = face.bbox
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2] - bbox[0])
            h = int(bbox[3] - bbox[1])
            # Bbox hợp lệ
            if w > 0 and h > 0:
                self.tracker.init(frame, (x, y, w, h))
                self.last_face = face
                self.frames_since_detect = 0
                return face

        self.tracker = None
        self.last_face = None
        return face

_global_tracker = FaceTracker(detect_interval=15)

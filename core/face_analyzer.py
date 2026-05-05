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
import insightface
from core.model_loader import build_cuda_provider_config

_FACE_ANALYSER: Optional[Any] = None
_ANALYSER_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    """Khởi tạo bộ phân tích mặt (Singleton + Double-Check Locking)."""
    global _FACE_ANALYSER
    if _FACE_ANALYSER is None:
        with _ANALYSER_LOCK:
            if _FACE_ANALYSER is None:
                providers = build_cuda_provider_config()
                _FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=providers,
                    allowed_modules=["detection", "recognition", "landmark_2d_106"],
                )
                _FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
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

# =============================================================================
# File: core/face_enhancer.py
# Mục đích: Làm nét khuôn mặt sau khi hoán đổi bằng mô hình GFPGAN.
#
# GIẢI THÍCH CHO BÁO CÁO:
#   Model inswapper_128 sinh ra khuôn mặt chỉ có 128x128 pixel.
#   Khi dán lên webcam HD (1080p), mặt sẽ bị mờ nhoè.
#   GFPGAN là mô hình AI Sinh tạo (Generative AI) chuyên phục hồi
#   và làm nét da, mắt, răng, tóc từ ảnh chất lượng thấp.
#
# Model: gfpgan-1024.onnx
#   - Input: [1, 3, 512, 512] float32, chuẩn hóa [-1, 1]
#   - Output: [1, 3, 1024, 1024] float32, chuẩn hóa [-1, 1]
# =============================================================================

import os
import threading
from typing import Optional
import cv2
import numpy as np
import onnxruntime

from core.model_loader import (
    build_provider_fallback_chain,
    get_models_directory,
    load_with_provider_fallback,
)
from core.face_analyzer import get_many_faces

_FACE_ENHANCER: Optional[onnxruntime.InferenceSession] = None
_FACE_ENHANCER_NAME: Optional[str] = None
_ENHANCER_LOCK = threading.Lock()

# Template 5 điểm mốc chuẩn FFHQ (khuôn mặt căn chỉnh theo chuẩn nghiên cứu)
FFHQ_TEMPLATE_512 = np.array([
    [192.98138, 239.94708],   # Mắt trái
    [318.90277, 240.19366],   # Mắt phải
    [256.63416, 314.01935],   # Mũi
    [201.26117, 371.41043],   # Khóe miệng trái
    [313.08905, 371.15118],   # Khóe miệng phải
], dtype=np.float32)


def get_face_enhancer() -> onnxruntime.InferenceSession:
    """Tải model làm nét lên bộ nhớ."""
    global _FACE_ENHANCER, _FACE_ENHANCER_NAME
    import config.globals as globals
    with _ENHANCER_LOCK:
        if _FACE_ENHANCER is None or _FACE_ENHANCER_NAME != globals.enhancer_model:
            model_path = os.path.join(get_models_directory(), globals.enhancer_model)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
            options = onnxruntime.SessionOptions()
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            _FACE_ENHANCER = load_with_provider_fallback(
                lambda providers: onnxruntime.InferenceSession(
                    model_path, sess_options=options, providers=providers
                ),
                build_provider_fallback_chain(),
                globals.enhancer_model,
            )
            _FACE_ENHANCER_NAME = globals.enhancer_model
    return _FACE_ENHANCER


def _align_face(frame: np.ndarray, landmarks_5: np.ndarray, size: int):
    """Căn chỉnh khuôn mặt theo template FFHQ chuẩn."""
    scale = size / 512.0
    template = FFHQ_TEMPLATE_512 * scale
    matrix, _ = cv2.estimateAffinePartial2D(landmarks_5, template, method=cv2.LMEDS)
    if matrix is None:
        return None, None
    aligned = cv2.warpAffine(
        frame, matrix, (size, size),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132)
    )
    return aligned, matrix


def _preprocess(face: np.ndarray) -> np.ndarray:
    """Chuyển ảnh BGR uint8 -> NCHW float32 [-1, 1] (chuẩn input GFPGAN)."""
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb / 255.0 - 0.5) / 0.5    # Chuẩn hóa về [-1, 1]
    chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(chw, axis=0)   # Thêm batch dimension -> NCHW


def _postprocess(output: np.ndarray) -> np.ndarray:
    """Chuyển output NCHW float32 [-1, 1] -> BGR uint8."""
    face = np.squeeze(output)            # Bỏ batch dim -> CHW
    face = np.transpose(face, (1, 2, 0)) # CHW -> HWC
    face = (face + 1.0) / 2.0 * 255.0   # [-1,1] -> [0,255]
    face = np.clip(face, 0, 255).astype(np.uint8)
    return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)


def _paste_back(frame: np.ndarray, enhanced: np.ndarray,
                matrix: np.ndarray, size: int) -> np.ndarray:
    """Dán khuôn mặt đã làm nét ngược lại lên khung hình gốc."""
    h, w = frame.shape[:2]
    inv_matrix = cv2.invertAffineTransform(matrix)
    inv_face = cv2.warpAffine(enhanced, inv_matrix, (w, h), borderValue=(0, 0, 0))

    # Tạo mask hình elip/tròn thay vì hình vuông để tránh lộ góc vuông
    face_mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    axes = (int(size * 0.45), int(size * 0.45))
    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1.0, -1)

    # Làm mờ viền (Gaussian Blur) với kernel lớn để viền siêu mềm
    blur_size = int(size * 0.2)
    if blur_size % 2 == 0:
        blur_size += 1
    face_mask = cv2.GaussianBlur(face_mask, (blur_size, blur_size), 0)

    # Giảm cường độ làm nét để tránh bị "giả" (Over-sharpening)
    import config.globals as globals
    enhancement_strength = globals.enhancement_strength
    face_mask = face_mask * enhancement_strength

    mask_3c = np.stack([face_mask] * 3, axis=-1)
    inv_mask = cv2.warpAffine(mask_3c, inv_matrix, (w, h), borderValue=(0, 0, 0))
    inv_mask = np.clip(inv_mask, 0.0, 1.0)

    result = frame.astype(np.float32) * (1.0 - inv_mask) + inv_face.astype(np.float32) * inv_mask
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Làm nét TẤT CẢ khuôn mặt trong khung hình.

    Quy trình cho MỖI khuôn mặt:
        1. Căn chỉnh (Align) mặt theo template FFHQ 512x512
        2. Tiền xử lý (Preprocess) -> tensor đầu vào cho GFPGAN
        3. Chạy suy luận (Inference) qua model GFPGAN
        4. Hậu xử lý (Postprocess) -> ảnh BGR uint8
        5. Dán lại (Paste back) lên khung hình gốc
    """
    session = get_face_enhancer()
    input_info = session.get_inputs()[0]
    input_name = input_info.name

    try:
        align_size = int(input_info.shape[2])
    except (ValueError, TypeError, IndexError):
        align_size = 512

    faces = get_many_faces(frame)
    if not faces:
        return frame

    result = frame.copy()
    for face in faces:
        if not hasattr(face, "kps") or face.kps is None:
            continue
        landmarks_5 = face.kps.astype(np.float32)
        if landmarks_5.shape[0] < 5:
            continue

        aligned, matrix = _align_face(frame, landmarks_5, align_size)
        if aligned is None:
            continue

        try:
            tensor = _preprocess(aligned)
            output = session.run(None, {input_name: tensor})[0]
            enhanced = _postprocess(output)

            # Resize nếu output lớn hơn input (GFPGAN 512->1024)
            eh, ew = enhanced.shape[:2]
            if eh != align_size or ew != align_size:
                enhanced = cv2.resize(enhanced, (align_size, align_size),
                                      interpolation=cv2.INTER_LANCZOS4)

            result = _paste_back(result, enhanced, matrix, align_size)
        except Exception as e:
            print(f"[FACE-ENHANCER] Lỗi: {e}")
            continue

    return result

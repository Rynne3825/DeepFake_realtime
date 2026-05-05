# =============================================================================
# File: core/face_swapper.py
# Mục đích: Hoán đổi khuôn mặt - Module cốt lõi nhất của hệ thống.
#
# Model sử dụng: inswapper_128.onnx
#   - Input: Ảnh mặt đích 128x128 + Embedding nguồn 512 chiều
#   - Output: Ảnh mặt 128x128 đã được tráo đổi
#   - Sau đó dùng Affine Transform để dán ngược lại lên khung hình gốc
# =============================================================================

import os
import threading
import warnings
from typing import Any, Optional
import cv2
import numpy as np
import insightface
from insightface.utils import face_align

from core.model_loader import build_cuda_provider_config, get_models_directory
from core.face_analyzer import get_one_face, get_many_faces
from core.face_masking import create_face_mask, create_mouth_mask
import config.globals as globals

_FACE_SWAPPER: Optional[Any] = None
_SWAPPER_LOCK = threading.Lock()


def get_face_swapper() -> Any:
    """Tải model hoán đổi khuôn mặt (inswapper_128.onnx) lên bộ nhớ."""
    global _FACE_SWAPPER
    with _SWAPPER_LOCK:
        if _FACE_SWAPPER is None:
            models_dir = get_models_directory()
            model_path = os.path.join(models_dir, "inswapper_128.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
            providers = build_cuda_provider_config()
            _FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, providers=providers
            )
    return _FACE_SWAPPER


def _paste_back_optimized(
    target_img: np.ndarray,
    swapped_face: np.ndarray,
    aligned_img: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    """
    Dán khuôn mặt đã swap ngược lại lên khung hình gốc (Optimized Paste Back).

    GIẢI THÍCH CHO BÁO CÁO:
        Sau khi model AI tạo ra khuôn mặt mới 128x128, ta cần:
        1. Đảo ngược phép biến đổi Affine để "xoay ngược" mặt về đúng góc.
        2. Tạo mặt nạ (mask) mềm mại ở viền để khi dán không bị lộ đường viền.
        3. Trộn (blend) khuôn mặt mới với ảnh gốc bằng Alpha Blending.

        Tối ưu: Chỉ xử lý vùng bounding box thay vì toàn bộ khung hình.
    """
    h, w = target_img.shape[:2]

    # Bước 1: Đảo ngược ma trận Affine
    inv_matrix = cv2.invertAffineTransform(affine_matrix)

    # Bước 2: Warp (biến dạng) khuôn mặt swap và mask về kích thước khung hình gốc
    face_warped = cv2.warpAffine(swapped_face, inv_matrix, (w, h), borderValue=0.0)
    mask_white = np.full(
        (aligned_img.shape[0], aligned_img.shape[1]), 255, dtype=np.float32
    )
    mask_warped = cv2.warpAffine(mask_white, inv_matrix, (w, h), borderValue=0.0)

    # Bước 3: Tìm bounding box của vùng mặt (để chỉ xử lý vùng cần thiết)
    rows = np.any(mask_warped > 20, axis=1)
    cols = np.any(mask_warped > 20, axis=0)
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    if len(row_idx) == 0 or len(col_idx) == 0:
        return target_img
    y1, y2 = row_idx[0], row_idx[-1]
    x1, x2 = col_idx[0], col_idx[-1]

    # Bước 4: Tính kích thước kernel cho Erosion và Blur
    mask_size = int(np.sqrt((y2 - y1) * (x2 - x1)))
    k_erode = max(mask_size // 10, 10)
    k_blur = max(mask_size // 20, 5)

    # Thêm padding cho vùng cắt
    pad = k_erode + k_blur + 2
    y1p, y2p = max(0, y1 - pad), min(h, y2 + pad + 1)
    x1p, x2p = max(0, x1 - pad), min(w, x2 + pad + 1)

    # Bước 5: Xử lý mask trên vùng cắt (crop) thay vì toàn bộ ảnh
    mask_crop = mask_warped[y1p:y2p, x1p:x2p]
    mask_crop[mask_crop > 20] = 255

    # Erosion: Thu nhỏ mask một chút để tránh lấn ra ngoài viền mặt
    kernel = np.ones((k_erode, k_erode), np.uint8)
    mask_crop = cv2.erode(mask_crop, kernel, iterations=1)

    # Gaussian Blur: Làm mềm viền mask
    blur_size = tuple(2 * i + 1 for i in (k_blur, k_blur))
    mask_crop = cv2.GaussianBlur(mask_crop, blur_size, 0)
    mask_crop /= 255.0

    # Bước 6: Alpha Blending - trộn khuôn mặt mới với ảnh gốc
    mask_3d = mask_crop[:, :, np.newaxis]
    fake_crop = face_warped[y1p:y2p, x1p:x2p].astype(np.float32)
    real_crop = target_img[y1p:y2p, x1p:x2p].astype(np.float32)
    blended = mask_3d * fake_crop + (1.0 - mask_3d) * real_crop

    result = target_img.copy()
    result[y1p:y2p, x1p:x2p] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _apply_face_mask_blend(
    original_frame: np.ndarray,
    swapped_frame: np.ndarray,
    face_mask: np.ndarray,
) -> np.ndarray:
    """Chỉ giữ lại vùng mặt đã swap trong phạm vi face mask."""
    if face_mask.ndim != 2:
        raise ValueError("face_mask phải là ảnh grayscale 2D")

    alpha = np.clip(face_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    alpha_3d = alpha[:, :, np.newaxis]
    blended = (
        swapped_frame.astype(np.float32) * alpha_3d
        + original_frame.astype(np.float32) * (1.0 - alpha_3d)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _restore_original_mouth(
    swapped_frame: np.ndarray,
    mouth_mask: np.ndarray,
    mouth_cutout: Optional[np.ndarray],
    mouth_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Dán lại vùng miệng gốc lên khung hình đã swap."""
    if mouth_cutout is None:
        return swapped_frame

    min_x, min_y, max_x, max_y = mouth_box
    if max_x <= min_x or max_y <= min_y:
        return swapped_frame

    result = swapped_frame.copy()
    alpha = mouth_mask[min_y:max_y, min_x:max_x].astype(np.float32) / 255.0
    alpha_3d = np.clip(alpha[:, :, np.newaxis], 0.0, 1.0)

    target_roi = result[min_y:max_y, min_x:max_x].astype(np.float32)
    source_roi = mouth_cutout.astype(np.float32)
    restored = source_roi * alpha_3d + target_roi * (1.0 - alpha_3d)
    result[min_y:max_y, min_x:max_x] = np.clip(restored, 0, 255).astype(np.uint8)
    return result


def swap_face(
    source_face: Any, target_face: Any, frame: np.ndarray
) -> np.ndarray:
    """
    Hoán đổi khuôn mặt nguồn lên khuôn mặt đích trong khung hình.

    Tham số:
        source_face: Khuôn mặt nguồn (từ ảnh do người dùng chọn).
        target_face: Khuôn mặt đích (phát hiện trên webcam).
        frame: Khung hình webcam gốc.

    Trả về:
        Khung hình đã được hoán đổi khuôn mặt.
    """
    swapper = get_face_swapper()
    if source_face is None or target_face is None:
        return frame

    try:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        # Gọi model AI: paste_back=False để ta tự xử lý việc dán lại
        swapped_face_128, affine_M = swapper.get(
            frame, target_face, source_face, paste_back=False
        )
        if swapped_face_128 is None:
            return frame

        # Lấy ảnh mặt đã căn chỉnh (aligned) để tạo mask
        # insightface/scikit-image currently emits a deprecation warning here.
        # Keep runtime logs clean while preserving the existing behavior.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            aligned_img, _ = face_align.norm_crop2(
                frame, target_face.kps, swapper.input_size[0]
            )

        # Dán khuôn mặt swap lại lên khung hình
        result = _paste_back_optimized(frame, swapped_face_128, aligned_img, affine_M)

        if globals.enable_masking:
            face_mask = create_face_mask(target_face, frame)
            result = _apply_face_mask_blend(frame, result, face_mask)

        if globals.mouth_mask:
            mouth_mask, mouth_cutout, mouth_box, _ = create_mouth_mask(target_face, frame)
            result = _restore_original_mouth(result, mouth_mask, mouth_cutout, mouth_box)

        # Áp dụng độ trong suốt (opacity) nếu cần
        opacity = globals.opacity
        if opacity < 1.0:
            original = frame.copy()
            result = cv2.addWeighted(original, 1 - opacity, result, opacity, 0)

        return np.clip(result, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"[FACE-SWAPPER] Lỗi: {e}")
        return frame


def process_frame(source_face: Any, frame: np.ndarray) -> np.ndarray:
    """
    Xử lý 1 khung hình hoàn chỉnh: phát hiện mặt đích -> hoán đổi.

    GIẢI THÍCH:
        Đây là hàm "điều phối" chính. Nó quyết định:
        - Nếu chế độ many_faces=True: Swap TẤT CẢ mặt trong frame.
        - Nếu many_faces=False: Chỉ swap 1 mặt chính (gần nhất bên trái).
    """
    if globals.many_faces:
        target_faces = get_many_faces(frame)
        if target_faces:
            for face in target_faces:
                frame = swap_face(source_face, face, frame)
    else:
        target_face = get_one_face(frame)
        if target_face:
            frame = swap_face(source_face, target_face, frame)

    return frame

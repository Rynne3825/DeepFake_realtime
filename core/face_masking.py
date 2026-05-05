# =============================================================================
# File: core/face_masking.py
# Mục đích: Tạo mặt nạ (Mask) để xử lý vùng che khuất trên khuôn mặt.
#
# GIẢI THÍCH CHO BÁO CÁO:
#   Khi hoán đổi khuôn mặt, nếu chỉ "dán đè" đơn giản thì khuôn mặt mới
#   sẽ đè lên cả lọn tóc, gọng kính, hoặc bàn tay đang che miệng.
#
#   Module này giải quyết vấn đề đó bằng cách:
#   1. Tạo "Mặt nạ khuôn mặt" (Face Mask) dựa trên 106 điểm mốc,
#      xác định chính xác vùng da mặt (được phép dán đè).
#   2. Tạo "Mặt nạ miệng" (Mouth Mask) để giữ nguyên biểu cảm miệng gốc,
#      tránh hiện tượng miệng bị "cứng" sau khi swap.
#
#   Thuật toán chính: Convex Hull + Gaussian Blur
#   - Convex Hull: Tìm đa giác lồi bao quanh các điểm mốc khuôn mặt.
#   - Gaussian Blur: Làm mềm viền mask để việc trộn (blend) trở nên tự nhiên.
# =============================================================================

import cv2
import numpy as np
from typing import Any, Tuple, Optional


def get_mouth_mask_expansion(mouth_mask_size: float) -> float:
    """Quy đổi thanh trượt 0-100 thành hệ số mở rộng vùng miệng."""
    return 1 + (mouth_mask_size / 100.0) * 2.5


def create_face_mask(face: Any, frame: np.ndarray) -> np.ndarray:
    """
    Tạo mặt nạ bao phủ vùng khuôn mặt dựa trên 106 điểm mốc.

    GIẢI THÍCH:
        1. Lấy 33 điểm mốc đầu tiên (viền khuôn mặt) từ landmark_2d_106.
        2. Tính Convex Hull (bao lồi) - đa giác nhỏ nhất bao quanh tất cả các điểm.
        3. Mở rộng (pad) bao lồi ra ngoài 5% để đảm bảo bao phủ hết khuôn mặt.
        4. Tô trắng (fill) vùng bên trong bao lồi -> đây là vùng "được phép swap".
        5. Gaussian Blur viền mask để chuyển tiếp mượt mà.

    Tham số:
        face: Đối tượng Face từ InsightFace (chứa landmark_2d_106).
        frame: Khung hình gốc (dùng để xác định kích thước mask).

    Trả về:
        np.ndarray: Mask 2D (grayscale), 255 = vùng mặt, 0 = vùng không phải mặt.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106

    if landmarks is None:
        return mask

    landmarks = landmarks.astype(np.int32)

    # Lấy 33 điểm viền khuôn mặt (từ trán xuống cằm, 2 bên má)
    face_outline = landmarks[0:33]

    # Tính Convex Hull (bao lồi)
    hull = cv2.convexHull(face_outline)

    # Mở rộng bao lồi ra ngoài 5% (padding)
    center = np.mean(face_outline, axis=0, dtype=np.float32)
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    directions = hull_pts - center
    norms = np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
    directions /= norms

    padding = int(np.linalg.norm(landmarks[0] - landmarks[16]) * 0.05)
    hull_padded = (hull_pts + directions * padding).astype(np.int32)

    # Tô trắng vùng khuôn mặt
    cv2.fillConvexPoly(mask, hull_padded, 255)

    # Làm mềm viền mask
    mask = cv2.GaussianBlur(mask, (5, 5), 3)

    return mask


def create_mouth_mask(
    face: Any, frame: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray], tuple, Optional[np.ndarray]]:
    """
    Tạo mặt nạ cho vùng miệng để giữ nguyên biểu cảm miệng gốc.

    GIẢI THÍCH CHO BÁO CÁO:
        Sau khi swap, miệng của khuôn mặt mới có thể không khớp với biểu cảm
        thực tế trên webcam (ví dụ: người dùng đang cười nhưng mặt swap lại
        có miệng đóng). Module này cắt vùng miệng GỐC (trước khi swap) và
        dán lại lên khuôn mặt đã swap, giữ nguyên biểu cảm miệng thật.

    Trả về:
        Tuple: (mask, mouth_cutout, mouth_box, polygon)
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None
    mouth_box = (0, 0, 0, 0)

    landmarks = face.landmark_2d_106
    if landmarks is None:
        return mask, mouth_cutout, mouth_box, lower_lip_polygon

    # Điểm mốc miệng: landmarks[52:72] (20 điểm bao quanh miệng)
    mouth_indices = list(range(52, 72))
    if max(mouth_indices) >= landmarks.shape[0]:
        return mask, mouth_cutout, mouth_box, lower_lip_polygon

    mouth_landmarks = landmarks[mouth_indices].astype(np.float32)
    center = np.mean(mouth_landmarks, axis=0)

    # Mở rộng vùng miệng theo cài đặt người dùng
    import config.globals as globals
    expansion = get_mouth_mask_expansion(globals.mouth_mask_size)

    offsets = mouth_landmarks - center
    expanded = mouth_landmarks.copy()
    expanded[:, 0] = center[0] + offsets[:, 0] * expansion
    expanded[:, 1] = center[1] + offsets[:, 1] * expansion
    expanded = expanded.astype(np.int32)

    # Tính bounding box
    min_x, min_y = np.min(expanded, axis=0)
    max_x, max_y = np.max(expanded, axis=0)
    padding = int((max_x - min_x) * 0.1)

    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(frame.shape[1], max_x + padding)
    max_y = min(frame.shape[0], max_y + padding)

    if max_x <= min_x or max_y <= min_y:
        return mask, mouth_cutout, mouth_box, lower_lip_polygon

    # Tạo mask cho vùng miệng
    mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    polygon_rel = expanded - [min_x, min_y]
    cv2.fillPoly(mask_roi, [polygon_rel], 255)
    mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 5)
    mask[min_y:max_y, min_x:max_x] = mask_roi

    # Cắt vùng miệng từ khung hình gốc
    mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
    lower_lip_polygon = expanded
    mouth_box = (min_x, min_y, max_x, max_y)

    return mask, mouth_cutout, mouth_box, lower_lip_polygon

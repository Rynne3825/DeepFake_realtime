# =============================================================================
# File: utils/image_utils.py
# Mục đích: Các hàm xử lý ảnh cơ bản dùng chung cho toàn bộ dự án.
#
# GIẢI THÍCH CHO BÁO CÁO:
#   Đây là "hộp công cụ" chứa các thao tác xử lý ảnh bằng thư viện OpenCV.
#   Mỗi hàm thực hiện một tác vụ nhỏ, được tái sử dụng ở nhiều nơi trong dự án.
# =============================================================================

import cv2
import numpy as np


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Thay đổi kích thước ảnh.
    
    Tham số:
        image: Ảnh đầu vào (mảng NumPy từ OpenCV).
        width: Chiều rộng mong muốn (đơn vị: pixel).
        height: Chiều cao mong muốn (đơn vị: pixel).
    
    Trả về:
        Ảnh đã được thay đổi kích thước.
    
    Ghi chú kỹ thuật:
        Sử dụng INTER_AREA khi thu nhỏ (chống răng cưa tốt hơn),
        INTER_LINEAR khi phóng to (nhanh và chất lượng đủ tốt).
    """
    h, w = image.shape[:2]

    if width * height < w * h:
        # Thu nhỏ ảnh -> dùng INTER_AREA để giữ chất lượng
        interpolation = cv2.INTER_AREA
    else:
        # Phóng to ảnh -> dùng INTER_LINEAR cho tốc độ nhanh
        interpolation = cv2.INTER_LINEAR

    return cv2.resize(image, (width, height), interpolation=interpolation)


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ảnh từ hệ màu BGR sang RGB.
    
    GIẢI THÍCH:
        OpenCV đọc ảnh theo thứ tự kênh màu BGR (Blue-Green-Red),
        nhưng hầu hết các mô hình AI và thư viện hiển thị (Tkinter, Pillow)
        lại sử dụng RGB (Red-Green-Blue). Hàm này thực hiện việc chuyển đổi đó.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def apply_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi màu sắc (Color Transfer) từ ảnh đích sang ảnh nguồn.
    
    GIẢI THÍCH CHO BÁO CÁO:
        Sau khi hoán đổi khuôn mặt, màu da của khuôn mặt mới có thể bị lệch
        so với phần cổ/tai của người gốc trên webcam (ví dụ: mặt trắng bệch
        trên thân người nâu). Kỹ thuật Color Transfer giúp cân bằng lại.
        
        Thuật toán hoạt động trong không gian màu LAB:
        - L (Lightness): Độ sáng
        - A: Trục màu xanh lá - đỏ
        - B: Trục màu xanh dương - vàng
        
        Bằng cách tính giá trị trung bình (mean) và độ lệch chuẩn (std) 
        của mỗi kênh, sau đó áp dụng công thức chuẩn hóa, màu sắc của 
        ảnh nguồn sẽ được điều chỉnh để phù hợp với ảnh đích.
    """
    # Chuyển sang float32 để tính toán chính xác
    source_f32 = source.astype(np.float32) / 255.0
    target_f32 = target.astype(np.float32) / 255.0

    # Chuyển từ BGR sang LAB (không gian màu phù hợp cho Color Transfer)
    source_lab = cv2.cvtColor(source_f32, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_f32, cv2.COLOR_BGR2LAB)

    # Tính giá trị thống kê (trung bình và độ lệch chuẩn) của mỗi kênh
    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    # Reshape để có thể thực hiện phép tính trên toàn bộ ảnh (Broadcasting)
    source_mean = source_mean.reshape(1, 1, 3).astype(np.float32)
    source_std = np.maximum(source_std.reshape(1, 1, 3), 1e-6).astype(np.float32)
    target_mean = target_mean.reshape(1, 1, 3).astype(np.float32)
    target_std = target_std.reshape(1, 1, 3).astype(np.float32)

    # Công thức Color Transfer:
    # result = (source - mean_source) * (std_target / std_source) + mean_target
    result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean

    # Chuyển ngược lại từ LAB sang BGR
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return np.clip(result_bgr * 255.0, 0, 255).astype(np.uint8)

# =============================================================================
# File: config/globals.py
# Mục đích: Lưu trữ các biến trạng thái TOÀN CỤC (Global State) của hệ thống.
#
# GIẢI THÍCH CHO BÁO CÁO:
#   Trong hệ thống thời gian thực (real-time), có 2 luồng (threads) chạy
#   song song:
#     1. Luồng Giao diện (UI Thread) - xử lý nút bấm, hiển thị hình ảnh.
#     2. Luồng Xử lý AI (Processing Thread) - đọc webcam, hoán đổi khuôn mặt.
#
#   Khi người dùng bấm nút "Bật Làm nét" trên giao diện, Luồng UI sẽ thay
#   đổi biến 'enable_enhancer' trong file này thành True. Luồng Xử lý AI
#   đọc biến này và tự động kích hoạt tính năng làm nét.
#
#   => File này đóng vai trò là "Bộ nhớ chung" (Shared Memory) giữa các luồng.
# =============================================================================

import sys
import threading

from config.performance_presets import DEFAULT_PRESET_KEY, apply_preset

# -----------------------------------------------------------------------------
# 1. ĐƯỜNG DẪN (Paths)
# -----------------------------------------------------------------------------
source_path: str | None = None      # Đường dẫn đến ảnh nguồn (ảnh chứa khuôn mặt muốn swap)
target_path: str | None = None      # Đường dẫn đến ảnh/video đích (nơi nhận khuôn mặt mới)

# -----------------------------------------------------------------------------
# 2. TRẠNG THÁI HỆ THỐNG (System State)
# -----------------------------------------------------------------------------
is_running: bool = False             # Hệ thống đang hoạt động hay không?
webcam_active: bool = False          # Webcam đang bật hay tắt?
show_fps: bool = True                # Hiển thị FPS trên giao diện?

# -----------------------------------------------------------------------------
# 3. TÙY CHỌN XỬ LÝ AI (AI Processing Options)
# -----------------------------------------------------------------------------
enable_swapper: bool = True          # Bật/tắt tính năng hoán đổi khuôn mặt
enable_enhancer: bool = False        # Bật/tắt tính năng làm nét (GFPGAN)
enhancement_strength: float = 0.45   # Cường độ làm nét (0.1 - 1.0)
enhancer_model: str = "GPEN-BFR-256.onnx" # Mô hình làm nét được sử dụng
enable_masking: bool = False         # Bật/tắt tính năng tạo mặt nạ chống lẹm
many_faces: bool = False             # Xử lý TẤT CẢ khuôn mặt hay chỉ 1 khuôn mặt?
live_mirror: bool = True             # Lật gương hình ảnh webcam (giống camera selfie)
quality_preset: str = DEFAULT_PRESET_KEY # Preset hiệu năng/chất lượng hiện tại
webcam_width: int = 960              # Độ phân giải webcam theo preset
webcam_height: int = 540             # Độ phân giải webcam theo preset
webcam_fps: int = 30                 # FPS webcam theo preset

# -----------------------------------------------------------------------------
# 4. CẤU HÌNH PHẦN CỨNG (Hardware Configuration)
# -----------------------------------------------------------------------------
# Danh sách Execution Provider (bộ thực thi) để ONNXRuntime biết chạy trên
# GPU (CUDA) hay CPU. Ví dụ: ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_providers: list[str] = []

# Khóa đồng bộ cho DirectML (chỉ cần trên Windows khi dùng AMD GPU)
dml_lock = threading.Lock()

# -----------------------------------------------------------------------------
# 5. THÔNG SỐ TINH CHỈNH (Fine-tuning Parameters)
# -----------------------------------------------------------------------------
opacity: float = 1.0                 # Độ trong suốt của khuôn mặt swap (0.0 = ẩn, 1.0 = hiện hoàn toàn)
mouth_mask: bool = False             # Bật/tắt giữ nguyên vùng miệng gốc
mouth_mask_size: float = 0.0        # Kích thước vùng miệng được giữ (0-100)

apply_preset(sys.modules[__name__], DEFAULT_PRESET_KEY)

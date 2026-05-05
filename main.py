# =============================================================================
# File: main.py
# Mục đích: Tệp khởi chạy chính (Entry Point) của hệ thống.
#
# GIẢI THÍCH CHO BÁO CÁO:
#   Đây là file "nhạc trưởng" (Orchestrator) của toàn bộ hệ thống.
#   Khi người dùng chạy "python main.py", file này thực hiện:
#     1. Thêm CUDA DLL vào PATH (để GPU NVIDIA hoạt động).
#     2. Phát hiện phần cứng và cấu hình Execution Provider.
#     3. Khởi chạy giao diện người dùng.
#
#   Lưu ý: Các mô hình AI KHÔNG được tải ở đây mà được tải theo kiểu
#   "Lazy Loading" (tải lười) - chỉ tải khi lần đầu tiên được gọi.
#   Điều này giúp ứng dụng khởi động nhanh hơn.
# =============================================================================

import os
import sys
import site
import warnings

# Fix encoding Unicode cho console Windows (để in emoji không bị lỗi)
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# insightface currently triggers a noisy deprecation warning from scikit-image.
# It does not affect runtime behavior of this app, so hide it from normal users.
warnings.filterwarnings(
    "ignore",
    message=r"`estimate` is deprecated since version 0\.26",
    category=FutureWarning,
)


def setup_cuda_path():
    """
    Thêm đường dẫn NVIDIA CUDA DLL vào biến môi trường PATH.
    
    GIẢI THÍCH:
        ONNXRuntime cần tìm các thư viện CUDA (.dll) để chạy trên GPU NVIDIA.
        Các file DLL này nằm trong thư mục site-packages/nvidia/ sau khi cài
        onnxruntime-gpu. Hàm này tự động tìm và thêm chúng vào PATH.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

    # Tìm thư mục site-packages
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    for root in (sys.prefix, sys.base_prefix):
        lib_site = os.path.join(root, "Lib", "site-packages")
        if os.path.isdir(lib_site):
            candidates.append(lib_site)

    # Thêm NVIDIA DLL vào PATH
    for site_pkg in candidates:
        nvidia_dir = os.path.join(site_pkg, "nvidia")
        if os.path.isdir(nvidia_dir):
            for pkg in os.listdir(nvidia_dir):
                bin_dir = os.path.join(nvidia_dir, pkg, "bin")
                if os.path.isdir(bin_dir):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]


def main():
    """Hàm khởi chạy chính."""
    print("=" * 60)
    print("  🎭 Hệ thống Hoán đổi Khuôn mặt Thời gian Thực")
    print("  📚 Đồ án: Deepfake Real-time trên Webcam")
    print("=" * 60)

    # Bước 1: Cấu hình CUDA
    print("\n[1/3] Đang cấu hình CUDA GPU...")
    setup_cuda_path()

    # Bước 2: Phát hiện phần cứng
    print("[2/3] Đang phát hiện phần cứng...")
    import config.globals as globals
    from core.model_loader import get_execution_providers

    globals.execution_providers = get_execution_providers()
    print(f"  → Execution Providers: {globals.execution_providers}")

    if "CUDAExecutionProvider" in globals.execution_providers:
        print("  → ✅ GPU NVIDIA (CUDA) đã được phát hiện!")
    else:
        print("  → ⚠️ Không tìm thấy GPU NVIDIA. Sử dụng CPU (sẽ chậm hơn).")

    # Bước 3: Khởi chạy giao diện
    print("[3/3] Đang khởi chạy giao diện...\n")
    from ui.app_window import DeepFakeApp

    app = DeepFakeApp()
    app.run()


if __name__ == "__main__":
    main()

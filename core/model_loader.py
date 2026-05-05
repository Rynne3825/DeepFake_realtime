# =============================================================================
# File: core/model_loader.py
# Mục đích: Quản lý việc tải các mô hình AI (Models) lên bộ nhớ GPU/CPU.
#
# GIẢI THÍCH CHO BÁO CÁO:
#   Trước khi hệ thống có thể hoán đổi khuôn mặt, các mô hình AI cần được
#   tải từ ổ cứng lên bộ nhớ (RAM hoặc VRAM của GPU).
#
#   File này quản lý việc:
#   1. Xác định phần cứng: Máy đang dùng GPU NVIDIA (CUDA) hay chỉ có CPU?
#   2. Cấu hình ONNXRuntime: Thiết lập tối ưu cho mỗi loại phần cứng.
#
#   ONNXRuntime là gì?
#   - Là công cụ của Microsoft giúp chạy các mô hình AI (.onnx) cực nhanh.
#   - Hỗ trợ nhiều loại GPU (NVIDIA, AMD, Apple Silicon).
#   - Nhanh hơn đáng kể so với PyTorch/TensorFlow khi chỉ cần suy luận (inference).
# =============================================================================

import os
import onnxruntime


def get_execution_providers() -> list:
    """
    Tự động phát hiện phần cứng và trả về danh sách Execution Provider tối ưu.
    
    GIẢI THÍCH:
        Execution Provider (EP) là "bộ thực thi" quyết định mô hình AI chạy ở đâu:
        - CUDAExecutionProvider: Chạy trên GPU NVIDIA (nhanh nhất).
        - DmlExecutionProvider: Chạy trên GPU AMD thông qua DirectML.
        - CPUExecutionProvider: Chạy trên CPU (chậm nhất, nhưng luôn khả dụng).
        
        Hàm này sẽ kiểm tra xem GPU NVIDIA có sẵn không. Nếu có, ưu tiên CUDA.
        Nếu không, tự động dùng CPU.
    
    Trả về:
        list: Danh sách EP theo thứ tự ưu tiên. Ví dụ:
              ['CUDAExecutionProvider', 'CPUExecutionProvider']
    """
    available = onnxruntime.get_available_providers()
    providers = []

    # Ưu tiên 1: GPU NVIDIA (CUDA) - nhanh nhất
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    # Ưu tiên 2: GPU AMD (DirectML) - Windows only
    if "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")

    # Luôn thêm CPU làm phương án dự phòng (fallback)
    providers.append("CPUExecutionProvider")

    return providers


def build_cuda_provider_config() -> list:
    """
    Xây dựng cấu hình tối ưu cho CUDA Execution Provider.
    
    GIẢI THÍCH CHO BÁO CÁO:
        Khi chạy trên GPU NVIDIA, ONNXRuntime có nhiều tùy chọn để tăng tốc.
        Cấu hình dưới đây là "Best Practice" (thực hành tốt nhất) được rút ra
        từ tài liệu chính thức của NVIDIA và ONNXRuntime:
        
        1. arena_extend_strategy = "kSameAsRequested":
           -> Tái sử dụng vùng nhớ GPU đã giải phóng thay vì cấp phát mới.
           
        2. cudnn_conv_algo_search = "EXHAUSTIVE":
           -> Thử nghiệm TẤT CẢ thuật toán tích chập (convolution) của cuDNN
              để tìm ra thuật toán nhanh nhất cho kiến trúc GPU cụ thể.
              (Chỉ chạy lần đầu, kết quả được cache lại).
           
        3. cudnn_conv_use_max_workspace = "1":
           -> Cho phép cuDNN dùng nhiều bộ nhớ hơn để đổi lấy tốc độ nhanh hơn.
           
        4. do_copy_in_default_stream = "0":
           -> Tách riêng luồng sao chép dữ liệu (Host ↔ GPU) khỏi luồng tính toán,
              cho phép chúng chạy song song (overlap) => tăng thông lượng (throughput).
    """
    providers = get_execution_providers()
    config = []

    for provider in providers:
        if provider == "CUDAExecutionProvider":
            config.append((
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "cudnn_conv_use_max_workspace": "1",
                    "do_copy_in_default_stream": "0",
                }
            ))
        else:
            config.append(provider)

    return config


def get_models_directory() -> str:
    """
    Trả về đường dẫn tuyệt đối đến thư mục chứa các file mô hình (.onnx).
    
    Cấu trúc:
        DeepFake/DeepFake_realtime/models/
        ├── inswapper_128.onnx       (Model hoán đổi khuôn mặt)
        ├── gfpgan-1024.onnx         (Model làm nét khuôn mặt)
        └── buffalo_l/               (Các model phân tích khuôn mặt)
    """
    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(app_root, "models")

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
import config.globals as globals


def _tensorrt_runtime_available() -> bool:
    """Return True if TensorRT runtime DLLs seem available on this machine.

    ONNX Runtime can expose the TensorRT EP even when TensorRT itself is not
    installed. On Windows this results in noisy errors like missing
    `nvinfer_10.dll`. We only enable TensorRT EP when the required DLLs are
    actually present on PATH.
    """
    path_value = os.environ.get("PATH", "")
    if not path_value:
        return False

    # Common TensorRT runtime DLL names across versions.
    required_any = {
        "nvinfer_10.dll",
        "nvinfer_9.dll",
        "nvinfer_8.dll",
        "nvinfer.dll",
    }

    for folder in path_value.split(os.pathsep):
        if not folder:
            continue
        try:
            for dll_name in required_any:
                if os.path.isfile(os.path.join(folder, dll_name)):
                    return True
        except OSError:
            continue

    return False


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

    # Ưu tiên 0: TensorRT - hiệu năng cao nhất trên GPU NVIDIA
    # Chỉ bật nếu TensorRT runtime DLL có sẵn, tránh spam lỗi thiếu nvinfer_*.dll.
    if "TensorrtExecutionProvider" in available and _tensorrt_runtime_available():
        providers.append("TensorrtExecutionProvider")

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
           
        4. do_copy_in_default_stream = "1":
           -> Giữ thao tác copy trong default stream để ổn định hơn với cấu hình
              runtime hiện tại của dự án.
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
                    "do_copy_in_default_stream": "1",
                }
            ))
        else:
            config.append(provider)

    return config


def build_provider_fallback_chain() -> list[list]:
    """Tạo danh sách cấu hình provider để thử lần lượt từ nhanh đến an toàn."""
    providers = get_execution_providers()
    chain = []

    # Cấu hình TensorRT (nếu có)
    if "TensorrtExecutionProvider" in providers:
        trt_config = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": os.path.abspath("./trt_cache"),
                    "trt_fp16_enable": True,
                }
            )
        ]
        if "CUDAExecutionProvider" in providers:
            trt_config.append((
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "cudnn_conv_use_max_workspace": "1",
                    "do_copy_in_default_stream": "1",
                }
            ))
        chain.append(trt_config)

    # Cấu hình CUDA
    primary = build_cuda_provider_config()
    if primary and primary not in chain:
        chain.append(primary)

    # Cấu hình CPU
    cpu_only = ["CPUExecutionProvider"]
    if cpu_only not in chain:
        chain.append(cpu_only)

    return chain


def load_with_provider_fallback(loader, provider_attempts: list[list], model_name: str):
    """
    Thử khởi tạo model/session theo nhiều cấu hình provider.

    Nếu GPU/CUDA lỗi do thiếu VRAM, paging file, hoặc provider DLL, hàm này
    sẽ tự fallback sang CPU thay vì làm sập toàn bộ ứng dụng ngay lần đầu.
    """
    last_error = None

    for providers in provider_attempts:
        try:
            result = loader(providers)
            runtime_providers = _extract_runtime_providers(result, providers)
            globals.runtime_providers[model_name] = runtime_providers
            print(f"[MODEL-LOADER] {model_name} -> providers: {runtime_providers}")
            return result
        except Exception as exc:
            last_error = exc
            print(f"[MODEL-LOADER] {model_name} failed with providers {providers}: {exc}")

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Không có provider nào để thử cho {model_name}")


def _extract_runtime_providers(result, attempted_providers: list) -> list[str]:
    if hasattr(result, "get_providers"):
        return list(result.get_providers())
    session = getattr(result, "session", None)
    if session is not None and hasattr(session, "get_providers"):
        return list(session.get_providers())
    return [
        provider[0] if isinstance(provider, tuple) else provider
        for provider in attempted_providers
    ]


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

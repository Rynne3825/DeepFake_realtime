from dataclasses import dataclass


@dataclass(frozen=True)
class BackendStatus:
    label: str
    color: str


def describe_backend(providers: list[str]) -> BackendStatus:
    if "CUDAExecutionProvider" in providers:
        return BackendStatus("AI Backend: GPU NVIDIA (CUDA)", "green")
    if "DmlExecutionProvider" in providers:
        return BackendStatus("AI Backend: GPU DirectML", "yellow")
    if "CPUExecutionProvider" in providers:
        return BackendStatus("AI Backend: CPU", "red")
    return BackendStatus("AI Backend: Chưa phát hiện", "yellow")


def describe_runtime_backend(runtime_providers: dict[str, list[str]]) -> BackendStatus:
    if not runtime_providers:
        return BackendStatus("AI Backend: Chưa tải model", "yellow")

    loaded = list(runtime_providers.values())
    all_cuda = all("CUDAExecutionProvider" in providers for providers in loaded)
    any_cuda = any("CUDAExecutionProvider" in providers for providers in loaded)
    any_cpu_only = any(
        "CUDAExecutionProvider" not in providers and "CPUExecutionProvider" in providers
        for providers in loaded
    )

    if all_cuda:
        return BackendStatus("AI Backend: GPU NVIDIA (runtime)", "green")
    if any_cuda and any_cpu_only:
        return BackendStatus("AI Backend: Mixed GPU/CPU", "yellow")
    if any_cpu_only:
        return BackendStatus("AI Backend: CPU (runtime)", "red")
    return BackendStatus("AI Backend: Runtime không rõ", "yellow")

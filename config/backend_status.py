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

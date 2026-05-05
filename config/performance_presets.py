from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerformancePreset:
    key: str
    label: str
    description: str
    webcam_width: int
    webcam_height: int
    webcam_fps: int
    enable_enhancer: bool
    enhancer_model: str
    enhancement_strength: float
    many_faces: bool = False


PRESETS: dict[str, PerformancePreset] = {
    "smooth": PerformancePreset(
        key="smooth",
        label="Mượt hơn",
        description="640x480, tắt làm nét để ưu tiên FPS ổn định.",
        webcam_width=640,
        webcam_height=480,
        webcam_fps=30,
        enable_enhancer=False,
        enhancer_model="GPEN-BFR-256.onnx",
        enhancement_strength=0.35,
    ),
    "balanced": PerformancePreset(
        key="balanced",
        label="Cân bằng",
        description="960x540, tắt làm nét mặc định để giữ độ mượt và bật khi cần.",
        webcam_width=960,
        webcam_height=540,
        webcam_fps=30,
        enable_enhancer=False,
        enhancer_model="GPEN-BFR-256.onnx",
        enhancement_strength=0.45,
    ),
    "quality": PerformancePreset(
        key="quality",
        label="Đẹp hơn",
        description="960x540, bật GFPGAN để ưu tiên chất lượng khuôn mặt.",
        webcam_width=960,
        webcam_height=540,
        webcam_fps=24,
        enable_enhancer=True,
        enhancer_model="gfpgan-1024.onnx",
        enhancement_strength=0.60,
    ),
}

DEFAULT_PRESET_KEY = "balanced"
LABEL_TO_KEY = {preset.label: key for key, preset in PRESETS.items()}
KEY_TO_LABEL = {key: preset.label for key, preset in PRESETS.items()}


def get_preset(key: str) -> PerformancePreset:
    try:
        return PRESETS[key]
    except KeyError as exc:
        raise ValueError(f"Preset không hợp lệ: {key}") from exc


def apply_preset(target: Any, key: str) -> PerformancePreset:
    preset = get_preset(key)
    target.quality_preset = preset.key
    target.webcam_width = preset.webcam_width
    target.webcam_height = preset.webcam_height
    target.webcam_fps = preset.webcam_fps
    target.enable_enhancer = preset.enable_enhancer
    target.enhancer_model = preset.enhancer_model
    target.enhancement_strength = preset.enhancement_strength
    target.many_faces = preset.many_faces
    return preset

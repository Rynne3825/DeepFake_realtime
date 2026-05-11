import unittest
from types import SimpleNamespace

from config.performance_presets import DEFAULT_PRESET_KEY, apply_preset, get_preset


class PerformancePresetTests(unittest.TestCase):
    def test_default_preset_is_balanced(self):
        preset = get_preset(DEFAULT_PRESET_KEY)

        self.assertEqual(DEFAULT_PRESET_KEY, "balanced")
        self.assertEqual(preset.label, "Cân bằng")
        self.assertEqual((preset.webcam_width, preset.webcam_height), (960, 540))
        self.assertFalse(preset.enable_enhancer)
        self.assertEqual(preset.enhancer_model, "GPEN-BFR-256.onnx")

    def test_apply_smooth_preset_updates_runtime_settings(self):
        runtime = SimpleNamespace(
            quality_preset="balanced",
            webcam_width=960,
            webcam_height=540,
            webcam_fps=30,
            enable_enhancer=True,
            enhancer_model="gfpgan-1024.onnx",
            enhancement_strength=0.6,
            many_faces=True,
        )

        apply_preset(runtime, "smooth")

        self.assertEqual(runtime.quality_preset, "smooth")
        self.assertEqual((runtime.webcam_width, runtime.webcam_height), (640, 480))
        self.assertEqual(runtime.webcam_fps, 30)
        self.assertFalse(runtime.enable_enhancer)
        self.assertEqual(runtime.enhancer_model, "GPEN-BFR-256.onnx")
        self.assertEqual(runtime.enhancement_strength, 0.35)
        self.assertFalse(runtime.many_faces)

    def test_unknown_preset_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_preset("ultra")


if __name__ == "__main__":
    unittest.main()

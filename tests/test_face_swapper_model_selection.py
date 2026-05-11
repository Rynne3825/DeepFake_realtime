import os
import tempfile
import unittest

import config.globals as globals

from core.face_swapper import resolve_inswapper_model_path


class FaceSwapperModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self._prev_use_fp16 = globals.use_fp16_inswapper

    def tearDown(self):
        globals.use_fp16_inswapper = self._prev_use_fp16

    def test_uses_fp32_model_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            globals.use_fp16_inswapper = False
            fp32_path = os.path.join(temp_dir, "inswapper_128.onnx")
            open(fp32_path, "wb").close()

            result = resolve_inswapper_model_path(temp_dir)

            self.assertEqual(result, fp32_path)

    def test_falls_back_to_fp16_when_fp32_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            globals.use_fp16_inswapper = False
            fp16_path = os.path.join(temp_dir, "inswapper_128_fp16.onnx")
            open(fp16_path, "wb").close()

            result = resolve_inswapper_model_path(temp_dir)

            self.assertEqual(result, fp16_path)

    def test_prefers_fp16_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            globals.use_fp16_inswapper = True
            fp32_path = os.path.join(temp_dir, "inswapper_128.onnx")
            fp16_path = os.path.join(temp_dir, "inswapper_128_fp16.onnx")
            open(fp32_path, "wb").close()
            open(fp16_path, "wb").close()

            result = resolve_inswapper_model_path(temp_dir)

            self.assertEqual(result, fp16_path)

    def test_falls_back_to_fp32_when_fp16_enabled_but_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            globals.use_fp16_inswapper = True
            fp32_path = os.path.join(temp_dir, "inswapper_128.onnx")
            open(fp32_path, "wb").close()

            result = resolve_inswapper_model_path(temp_dir)

            self.assertEqual(result, fp32_path)

    def test_raises_clear_error_when_no_supported_model_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            globals.use_fp16_inswapper = False
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_inswapper_model_path(temp_dir)

            message = str(ctx.exception)
            self.assertIn("inswapper_128.onnx", message)


if __name__ == "__main__":
    unittest.main()

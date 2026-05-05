import os
import tempfile
import unittest

from core.face_swapper import resolve_inswapper_model_path


class FaceSwapperModelSelectionTests(unittest.TestCase):
    def test_prefers_fp32_model_when_both_files_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fp32_path = os.path.join(temp_dir, "inswapper_128.onnx")
            fp16_path = os.path.join(temp_dir, "inswapper_128_fp16.onnx")
            open(fp32_path, "wb").close()
            open(fp16_path, "wb").close()

            result = resolve_inswapper_model_path(temp_dir)

            self.assertEqual(result, fp32_path)

    def test_falls_back_to_fp16_model_when_fp32_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fp16_path = os.path.join(temp_dir, "inswapper_128_fp16.onnx")
            open(fp16_path, "wb").close()

            result = resolve_inswapper_model_path(temp_dir)

            self.assertEqual(result, fp16_path)

    def test_raises_clear_error_when_no_supported_model_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_inswapper_model_path(temp_dir)

            message = str(ctx.exception)
            self.assertIn("inswapper_128.onnx", message)
            self.assertIn("inswapper_128_fp16.onnx", message)


if __name__ == "__main__":
    unittest.main()

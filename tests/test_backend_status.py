import unittest

from config.backend_status import describe_backend


class BackendStatusTests(unittest.TestCase):
    def test_cuda_provider_reports_nvidia_gpu(self):
        status = describe_backend(["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.assertEqual(status.label, "AI Backend: GPU NVIDIA (CUDA)")
        self.assertEqual(status.color, "green")

    def test_cpu_only_reports_cpu_mode(self):
        status = describe_backend(["CPUExecutionProvider"])

        self.assertEqual(status.label, "AI Backend: CPU")
        self.assertEqual(status.color, "red")

    def test_empty_provider_list_reports_unknown_state(self):
        status = describe_backend([])

        self.assertEqual(status.label, "AI Backend: Chưa phát hiện")
        self.assertEqual(status.color, "yellow")


if __name__ == "__main__":
    unittest.main()

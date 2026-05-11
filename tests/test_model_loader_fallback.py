import unittest

from core.model_loader import load_with_provider_fallback


class ModelLoaderFallbackTests(unittest.TestCase):
    def test_returns_first_successful_loader_result(self):
        calls = []

        def loader(providers):
            calls.append(providers)
            if providers == ["GPU"]:
                raise RuntimeError("gpu failed")
            return {"providers": providers}

        result = load_with_provider_fallback(
            loader,
            [["GPU"], ["CPU"]],
            "test-model",
        )

        self.assertEqual(result, {"providers": ["CPU"]})
        self.assertEqual(calls, [["GPU"], ["CPU"]])

    def test_raises_last_error_when_all_attempts_fail(self):
        def loader(providers):
            raise RuntimeError(f"failed:{providers[0]}")

        with self.assertRaises(RuntimeError) as ctx:
            load_with_provider_fallback(loader, [["GPU"], ["CPU"]], "test-model")

        self.assertEqual(str(ctx.exception), "failed:CPU")


if __name__ == "__main__":
    unittest.main()

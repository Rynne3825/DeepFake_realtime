import unittest

from core.face_masking import get_mouth_mask_expansion


class FaceMaskingSettingsTests(unittest.TestCase):
    def test_mouth_mask_expansion_scales_from_percentage(self):
        self.assertEqual(get_mouth_mask_expansion(0.0), 1.0)
        self.assertEqual(get_mouth_mask_expansion(20.0), 1.5)
        self.assertEqual(get_mouth_mask_expansion(100.0), 3.5)


if __name__ == "__main__":
    unittest.main()

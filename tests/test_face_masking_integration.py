import unittest

import numpy as np

from core.face_swapper import _apply_face_mask_blend, _restore_original_mouth


class FaceMaskingIntegrationTests(unittest.TestCase):
    def test_apply_face_mask_blend_only_changes_masked_region(self):
        original = np.zeros((3, 3, 3), dtype=np.uint8)
        swapped = np.full((3, 3, 3), 200, dtype=np.uint8)
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask[1, 1] = 255

        result = _apply_face_mask_blend(original, swapped, mask)

        self.assertEqual(int(result[1, 1, 0]), 200)
        self.assertTrue(np.array_equal(result[0, 0], original[0, 0]))
        self.assertTrue(np.array_equal(result[2, 2], original[2, 2]))

    def test_restore_original_mouth_reinserts_cutout_inside_box(self):
        swapped = np.full((4, 4, 3), 150, dtype=np.uint8)
        mouth_cutout = np.zeros((2, 2, 3), dtype=np.uint8)
        mouth_cutout[:, :] = [10, 20, 30]
        mouth_mask = np.zeros((4, 4), dtype=np.uint8)
        mouth_mask[1:3, 1:3] = 255

        result = _restore_original_mouth(
            swapped,
            mouth_mask,
            mouth_cutout,
            (1, 1, 3, 3),
        )

        self.assertTrue(np.array_equal(result[1, 1], [10, 20, 30]))
        self.assertTrue(np.array_equal(result[2, 2], [10, 20, 30]))
        self.assertTrue(np.array_equal(result[0, 0], [150, 150, 150]))


if __name__ == "__main__":
    unittest.main()

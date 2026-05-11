import unittest
from unittest.mock import patch

import numpy as np

import config.globals as globals
from core.face_swapper import process_frame


class FaceSwapperProcessFrameTests(unittest.TestCase):
    def setUp(self):
        self._prev_many_faces = globals.many_faces

    def tearDown(self):
        globals.many_faces = self._prev_many_faces

    def test_many_faces_mode_swaps_each_detected_face(self):
        globals.many_faces = True
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        source_face = object()
        faces = ["face-a", "face-b"]

        with patch("core.face_swapper.get_many_faces", return_value=faces), patch(
            "core.face_swapper.swap_face",
            side_effect=lambda src, face, current_frame: current_frame + 1,
        ) as swap_mock:
            result = process_frame(source_face, frame)

        self.assertEqual(swap_mock.call_count, 2)
        self.assertTrue(np.array_equal(result, np.full((2, 2, 3), 2, dtype=np.uint8)))

    def test_single_face_mode_uses_tracker_result(self):
        globals.many_faces = False
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        source_face = object()
        tracked_face = object()

        with patch("core.face_swapper._global_tracker.update", return_value=tracked_face) as tracker_mock, patch(
            "core.face_swapper.swap_face",
            side_effect=lambda src, face, current_frame: current_frame + 3,
        ) as swap_mock:
            result = process_frame(source_face, frame)

        tracker_mock.assert_called_once()
        swap_mock.assert_called_once_with(source_face, tracked_face, frame)
        self.assertTrue(np.array_equal(result, np.full((2, 2, 3), 3, dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()

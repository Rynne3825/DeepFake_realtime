import unittest
from types import SimpleNamespace

import numpy as np

from core.face_analyzer import FaceTracker


class _StubTracker:
    def __init__(self, bbox):
        self._bbox = bbox

    def update(self, frame):
        return True, self._bbox


class FaceTrackerTests(unittest.TestCase):
    def test_tracking_updates_landmark_106_with_bbox_motion(self):
        tracker = FaceTracker(detect_interval=15)
        tracker.tracker = _StubTracker((20.0, 30.0, 20.0, 20.0))
        tracker.frames_since_detect = 0
        tracker.last_face = SimpleNamespace(
            bbox=np.array([10.0, 10.0, 20.0, 20.0], dtype=np.float32),
            kps=np.array(
                [
                    [12.0, 12.0],
                    [18.0, 12.0],
                    [15.0, 15.0],
                    [13.0, 18.0],
                    [17.0, 18.0],
                ],
                dtype=np.float32,
            ),
            landmark_2d_106=np.array(
                [
                    [11.0, 11.0],
                    [15.0, 13.0],
                    [19.0, 19.0],
                ],
                dtype=np.float32,
            ),
        )

        result = tracker.update(np.zeros((4, 4, 3), dtype=np.uint8), analyzer_func=None)

        self.assertIs(result, tracker.last_face)
        np.testing.assert_allclose(
            result.landmark_2d_106,
            np.array(
                [
                    [22.0, 32.0],
                    [30.0, 36.0],
                    [38.0, 48.0],
                ],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()

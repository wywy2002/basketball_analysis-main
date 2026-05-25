import os
import tempfile
import unittest

import numpy as np

from pose_estimator.player_pose_2d_estimator import POSE_2D_SCHEMA_VERSION
from pose_estimator.pose_3d_lifter import Pose3DLifter
from utils import read_stub


class FakePose3DLifter(Pose3DLifter):
    def __init__(self, min_sequence_length=9):
        self.model_path = None
        self.min_sequence_length = min_sequence_length
        self.min_valid_keypoints = 8
        self.keypoint_conf_threshold = 0.20
        self.max_interpolation_gap = 2
        self.model = object()
        self.torch = None
        self.last_input_shape = None

    def _predict_3d(self, normalized_keypoints):
        self.last_input_shape = normalized_keypoints.shape
        prediction = np.zeros((len(normalized_keypoints), 17, 3), dtype=np.float32)
        prediction[..., :2] = normalized_keypoints
        prediction[..., 2] = 1.0
        return prediction


def make_frames(count, height=100, width=200):
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)]


def make_pose_tracks(count, track_id=7):
    keypoints = [[float(index), float(index + 1)] for index in range(17)]
    return [
        {
            track_id: {
                "keypoints_2d": keypoints,
                "confidence": [1.0] * 17,
                "valid_keypoints": 17,
                "schema_version": POSE_2D_SCHEMA_VERSION,
            }
        }
        for _ in range(count)
    ]


class Pose3DLifterTests(unittest.TestCase):
    def test_generates_and_saves_3d_pose_stub_for_valid_sequence(self):
        frames = make_frames(10)
        pose_2d_tracks = make_pose_tracks(10)
        lifter = FakePose3DLifter()

        with tempfile.TemporaryDirectory() as tmpdir:
            stub_path = os.path.join(tmpdir, "pose_3d_stubs.pkl")
            pose_3d_tracks = lifter.get_pose_tracks(frames, pose_2d_tracks, False, stub_path)
            cached = read_stub(True, stub_path)

        self.assertEqual(len(pose_3d_tracks), 10)
        self.assertEqual(lifter.last_input_shape, (10, 17, 2))
        self.assertIn(7, pose_3d_tracks[0])
        self.assertEqual(np.asarray(pose_3d_tracks[0][7]["keypoints_3d"]).shape, (17, 3))
        self.assertEqual(pose_3d_tracks[0][7]["keypoints_2d"], pose_2d_tracks[0][7]["keypoints_2d"])
        self.assertEqual(pose_3d_tracks[0][7]["confidence"], pose_2d_tracks[0][7]["confidence"])
        self.assertEqual(cached, pose_3d_tracks)

    def test_skips_segments_shorter_than_min_sequence_length(self):
        frames = make_frames(8)
        pose_2d_tracks = make_pose_tracks(8)

        pose_3d_tracks = FakePose3DLifter().get_pose_tracks(frames, pose_2d_tracks)

        self.assertEqual(pose_3d_tracks, [{} for _ in frames])

    def test_missing_model_does_not_write_empty_3d_stub(self):
        frames = make_frames(10)
        pose_2d_tracks = make_pose_tracks(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            stub_path = os.path.join(tmpdir, "pose_3d_stubs.pkl")
            pose_3d_tracks = Pose3DLifter(os.path.join(tmpdir, "missing.pt")).get_pose_tracks(
                frames,
                pose_2d_tracks,
                False,
                stub_path,
            )

            self.assertEqual(pose_3d_tracks, [{} for _ in frames])
            self.assertFalse(os.path.exists(stub_path))

    def test_invalid_2d_keypoint_shapes_are_ignored(self):
        frames = make_frames(10)
        pose_2d_tracks = make_pose_tracks(10)
        pose_2d_tracks[0][7]["keypoints_2d"] = [[0.0, 0.0]]

        pose_3d_tracks = FakePose3DLifter().get_pose_tracks(frames, pose_2d_tracks)

        self.assertEqual(pose_3d_tracks[0], {})
        self.assertIn(7, pose_3d_tracks[1])

    def test_interpolates_short_missing_2d_gap(self):
        frames = make_frames(10)
        pose_2d_tracks = make_pose_tracks(10)
        pose_2d_tracks[4] = {}
        lifter = FakePose3DLifter()

        pose_3d_tracks = lifter.get_pose_tracks(frames, pose_2d_tracks)

        self.assertIn(7, pose_3d_tracks[4])
        repaired = np.asarray(pose_3d_tracks[4][7]["keypoints_2d"], dtype=np.float32)
        expected = np.asarray(pose_2d_tracks[3][7]["keypoints_2d"], dtype=np.float32)
        self.assertTrue(np.allclose(repaired, expected))

    def test_skips_large_missing_2d_gap(self):
        frames = make_frames(10)
        pose_2d_tracks = make_pose_tracks(10)
        pose_2d_tracks[3] = {}
        pose_2d_tracks[4] = {}
        pose_2d_tracks[5] = {}

        pose_3d_tracks = FakePose3DLifter().get_pose_tracks(frames, pose_2d_tracks)

        self.assertEqual(pose_3d_tracks, [{} for _ in frames])

    def test_legacy_2d_pose_records_are_ignored(self):
        frames = make_frames(10)
        pose_2d_tracks = make_pose_tracks(10)
        for frame in pose_2d_tracks:
            frame[7].pop("schema_version")

        pose_3d_tracks = FakePose3DLifter().get_pose_tracks(frames, pose_2d_tracks)

        self.assertEqual(pose_3d_tracks, [{} for _ in frames])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from pose_estimator.player_pose_2d_estimator import (
    POSE_2D_SCHEMA_VERSION,
    PlayerPose2DEstimator,
)


class ArrayLike:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeKeypoints:
    def __init__(self, xy, conf=None):
        self.xy = ArrayLike(xy)
        self.conf = ArrayLike(conf) if conf is not None else None


class FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = ArrayLike(xyxy)


class FakeResult:
    def __init__(self, xy, conf=None, boxes=None):
        self.keypoints = FakeKeypoints(xy, conf)
        self.boxes = FakeBoxes(boxes) if boxes is not None else None


class FakeModel:
    def __init__(self, results):
        self.results = list(results)

    def predict(self, inputs, conf=0.25, verbose=False):
        count = len(inputs)
        output = self.results[:count]
        self.results = self.results[count:]
        return output


def make_estimator(model=None, enable_crop_fallback=True):
    estimator = PlayerPose2DEstimator.__new__(PlayerPose2DEstimator)
    estimator.model = model or FakeModel([])
    estimator.conf = 0.25
    estimator.padding = 0.30
    estimator.batch_size = 16
    estimator.match_threshold = 0.25
    estimator.min_valid_keypoints = 8
    estimator.enable_crop_fallback = enable_crop_fallback
    estimator.keypoint_conf_threshold = 0.20
    estimator.max_outside_bbox_ratio = 0.45
    estimator.joint_filter_padding = 0.25
    estimator.last_stats = estimator._new_stats()
    return estimator


def make_keypoints(x1, y1, x2, y2):
    xs = np.linspace(x1 + 5, x2 - 5, 17, dtype=np.float32)
    ys = np.linspace(y1 + 5, y2 - 5, 17, dtype=np.float32)
    return np.stack([xs, ys], axis=1)


class PlayerPose2DEstimatorTests(unittest.TestCase):
    def test_selects_pose_matching_player_bbox_over_highest_confidence(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        player_tracks = {1: {"bbox": [10.0, 10.0, 60.0, 110.0]}}
        far_pose = make_keypoints(120, 10, 170, 110)
        matching_pose = make_keypoints(10, 10, 60, 110)
        result = FakeResult(
            [far_pose, matching_pose],
            [np.full(17, 0.99), np.full(17, 0.70)],
            [[120, 10, 170, 110], [10, 10, 60, 110]],
        )
        estimator = make_estimator(enable_crop_fallback=False)

        frame_poses = estimator._estimate_frame_poses(frame, player_tracks, result)

        self.assertIn(1, frame_poses)
        self.assertEqual(frame_poses[1]["source"], "full_frame")
        self.assertTrue(np.allclose(frame_poses[1]["keypoints_2d"], matching_pose))
        self.assertEqual(frame_poses[1]["schema_version"], POSE_2D_SCHEMA_VERSION)

    def test_rejects_low_quality_candidates(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        player_tracks = {1: {"bbox": [10.0, 10.0, 60.0, 110.0]}}
        keypoints = make_keypoints(10, 10, 60, 110)
        confidence = np.array([0.90] * 4 + [0.01] * 13, dtype=np.float32)
        result = FakeResult([keypoints], [confidence], [[10, 10, 60, 110]])
        estimator = make_estimator(enable_crop_fallback=False)

        frame_poses = estimator._estimate_frame_poses(frame, player_tracks, result)

        self.assertEqual(frame_poses, {})

    def test_one_candidate_is_assigned_to_only_one_track(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        player_tracks = {
            1: {"bbox": [10.0, 10.0, 60.0, 110.0]},
            2: {"bbox": [12.0, 12.0, 62.0, 112.0]},
        }
        keypoints = make_keypoints(10, 10, 60, 110)
        result = FakeResult(
            [keypoints],
            [np.full(17, 0.90)],
            [[10, 10, 60, 110]],
        )
        estimator = make_estimator(enable_crop_fallback=False)

        frame_poses = estimator._estimate_frame_poses(frame, player_tracks, result)

        self.assertEqual(len(frame_poses), 1)

    def test_crop_fallback_maps_keypoints_back_to_full_frame(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        player_tracks = {1: {"bbox": [50.0, 50.0, 100.0, 150.0]}}
        empty_full_frame_result = FakeResult([], [], [])
        local_pose = make_keypoints(20, 35, 70, 135)
        crop_result = FakeResult(
            [local_pose],
            [np.full(17, 0.90)],
            [[20, 35, 70, 135]],
        )
        estimator = make_estimator(
            model=FakeModel([crop_result]),
            enable_crop_fallback=True,
        )

        frame_poses = estimator._estimate_frame_poses(
            frame,
            player_tracks,
            empty_full_frame_result,
        )

        self.assertIn(1, frame_poses)
        self.assertEqual(frame_poses[1]["source"], "crop")
        first_keypoint = frame_poses[1]["keypoints_2d"][0]
        self.assertGreaterEqual(first_keypoint[0], 50.0)
        self.assertGreaterEqual(first_keypoint[1], 50.0)

    def test_legacy_cache_is_invalid(self):
        legacy_cache = [
            {
                1: {
                    "bbox": [10.0, 10.0, 60.0, 110.0],
                    "keypoints_2d": make_keypoints(10, 10, 60, 110).tolist(),
                    "confidence": [1.0] * 17,
                }
            }
        ]

        self.assertFalse(PlayerPose2DEstimator.is_valid_pose_cache(legacy_cache, 1))

    def test_outlier_keypoints_are_removed_before_saving_pose(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        player_tracks = {1: {"bbox": [50.0, 50.0, 110.0, 170.0]}}
        keypoints = make_keypoints(50, 50, 110, 170)
        keypoints[9] = [70.0, 5.0]
        result = FakeResult(
            [keypoints],
            [np.full(17, 0.90)],
            [[50, 50, 110, 170]],
        )
        estimator = make_estimator(enable_crop_fallback=False)

        frame_poses = estimator._estimate_frame_poses(frame, player_tracks, result)

        self.assertIn(1, frame_poses)
        self.assertEqual(frame_poses[1]["keypoints_2d"][9], [0.0, 0.0])
        self.assertEqual(frame_poses[1]["confidence"][9], 0.0)
        self.assertEqual(frame_poses[1]["valid_keypoints"], 16)


if __name__ == "__main__":
    unittest.main()

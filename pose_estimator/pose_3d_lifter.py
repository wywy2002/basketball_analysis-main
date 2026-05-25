import os

import numpy as np

from pose_estimator.player_pose_2d_estimator import (
    COCO_KEYPOINT_COUNT,
    PlayerPose2DEstimator,
)
from utils import read_stub, save_stub


class Pose3DLifter:
    """
    Runs an optional TorchScript 2D-to-3D lifting model.

    Expected model contract:
    - input: torch.Tensor shaped (1, frames, 17, 2), normalized screen coordinates
    - output: torch.Tensor shaped (1, frames, 17, 3) or (frames, 17, 3)
    """

    def __init__(
        self,
        model_path,
        min_sequence_length=9,
        min_valid_keypoints=8,
        keypoint_conf_threshold=0.20,
        max_interpolation_gap=2,
    ):
        self.model_path = model_path
        self.min_sequence_length = min_sequence_length
        self.min_valid_keypoints = min_valid_keypoints
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.max_interpolation_gap = max_interpolation_gap
        self.model = None

        if model_path and os.path.exists(model_path):
            import torch

            self.torch = torch
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
        else:
            self.torch = None

    @property
    def is_available(self):
        return self.model is not None

    def get_pose_tracks(self, frames, pose_2d_tracks, read_from_stub=False, stub_path=None):
        pose_tracks = read_stub(read_from_stub, stub_path)
        if pose_tracks is not None and len(pose_tracks) == len(frames):
            return pose_tracks

        if not frames:
            pose_3d_tracks = []
            if self.is_available and stub_path is not None:
                save_stub(stub_path, pose_3d_tracks)
            return pose_3d_tracks

        if not self.is_available:
            return [{} for _ in frames]

        frame_h, frame_w = frames[0].shape[:2]
        pose_3d_tracks = [{} for _ in frames]
        track_sequences = self._collect_track_sequences(pose_2d_tracks)

        for track_id, sequence in track_sequences.items():
            for segment in self._split_interpolatable_segments(sequence):
                repaired = self._build_repaired_segment(segment)
                if repaired is None or len(repaired["frame_indices"]) < self.min_sequence_length:
                    continue

                normalized = self._normalize_screen_coordinates(
                    repaired["keypoints"],
                    frame_w,
                    frame_h,
                )
                predicted = self._predict_3d(normalized)

                for local_idx, frame_idx in enumerate(repaired["frame_indices"]):
                    pose_3d_tracks[frame_idx][track_id] = {
                        "keypoints_3d": predicted[local_idx].astype(float).tolist(),
                        "keypoints_2d": repaired["keypoints"][local_idx].astype(float).tolist(),
                        "confidence": repaired["confidence"][local_idx].astype(float).tolist(),
                    }

        if stub_path is not None:
            save_stub(stub_path, pose_3d_tracks)
        return pose_3d_tracks

    def _collect_track_sequences(self, pose_2d_tracks):
        sequences = {}
        for frame_idx, frame_poses in enumerate(pose_2d_tracks):
            for track_id, pose in frame_poses.items():
                if not PlayerPose2DEstimator.is_valid_pose_record(
                    pose,
                    self.min_valid_keypoints,
                ):
                    continue

                keypoints = np.asarray(pose.get("keypoints_2d"), dtype=np.float32)
                confidence = np.asarray(pose.get("confidence"), dtype=np.float32)
                if self._valid_joint_count(keypoints, confidence) < self.min_valid_keypoints:
                    continue
                sequences.setdefault(track_id, []).append(
                    {
                        "frame_idx": frame_idx,
                        "keypoints": keypoints,
                        "confidence": confidence,
                    }
                )
        return sequences

    def _split_interpolatable_segments(self, sequence):
        if not sequence:
            return []

        sequence = sorted(sequence, key=lambda item: item["frame_idx"])
        segments = []
        current = [sequence[0]]
        for item in sequence[1:]:
            gap = item["frame_idx"] - current[-1]["frame_idx"]
            if gap <= self.max_interpolation_gap + 1:
                current.append(item)
            else:
                segments.append(current)
                current = [item]
        segments.append(current)
        return segments

    def _build_repaired_segment(self, segment):
        start_frame = segment[0]["frame_idx"]
        end_frame = segment[-1]["frame_idx"]
        frame_indices = list(range(start_frame, end_frame + 1))
        frame_to_pose = {item["frame_idx"]: item for item in segment}

        keypoints = np.zeros((len(frame_indices), COCO_KEYPOINT_COUNT, 2), dtype=np.float32)
        confidence = np.zeros((len(frame_indices), COCO_KEYPOINT_COUNT), dtype=np.float32)

        for local_idx, frame_idx in enumerate(frame_indices):
            if frame_idx in frame_to_pose:
                keypoints[local_idx] = frame_to_pose[frame_idx]["keypoints"]
                confidence[local_idx] = frame_to_pose[frame_idx]["confidence"]
                continue

            previous_pose, next_pose = self._nearest_known_poses(frame_idx, segment)
            if previous_pose is None or next_pose is None:
                return None
            gap = next_pose["frame_idx"] - previous_pose["frame_idx"] - 1
            if gap > self.max_interpolation_gap:
                return None

            alpha = (
                (frame_idx - previous_pose["frame_idx"])
                / (next_pose["frame_idx"] - previous_pose["frame_idx"])
            )
            keypoints[local_idx] = (
                previous_pose["keypoints"] * (1.0 - alpha)
                + next_pose["keypoints"] * alpha
            )
            confidence[local_idx] = np.minimum(
                previous_pose["confidence"],
                next_pose["confidence"],
            ) * 0.75

        self._repair_low_confidence_single_points(keypoints, confidence)
        if not self._segment_has_enough_valid_keypoints(keypoints, confidence):
            return None

        return {
            "frame_indices": frame_indices,
            "keypoints": keypoints,
            "confidence": confidence,
        }

    def _nearest_known_poses(self, frame_idx, segment):
        previous_pose = None
        next_pose = None
        for item in segment:
            if item["frame_idx"] < frame_idx:
                previous_pose = item
            elif item["frame_idx"] > frame_idx:
                next_pose = item
                break
        return previous_pose, next_pose

    def _repair_low_confidence_single_points(self, keypoints, confidence):
        for frame_idx in range(1, len(keypoints) - 1):
            for joint_idx in range(COCO_KEYPOINT_COUNT):
                if confidence[frame_idx, joint_idx] >= self.keypoint_conf_threshold:
                    continue
                prev_conf = confidence[frame_idx - 1, joint_idx]
                next_conf = confidence[frame_idx + 1, joint_idx]
                if (
                    prev_conf >= self.keypoint_conf_threshold
                    and next_conf >= self.keypoint_conf_threshold
                ):
                    keypoints[frame_idx, joint_idx] = (
                        keypoints[frame_idx - 1, joint_idx]
                        + keypoints[frame_idx + 1, joint_idx]
                    ) / 2.0
                    confidence[frame_idx, joint_idx] = min(prev_conf, next_conf) * 0.75

    def _segment_has_enough_valid_keypoints(self, keypoints, confidence):
        for frame_idx in range(len(keypoints)):
            if self._valid_joint_count(keypoints[frame_idx], confidence[frame_idx]) < self.min_valid_keypoints:
                return False
        return True

    def _valid_joint_count(self, keypoints, confidence):
        if keypoints.shape != (COCO_KEYPOINT_COUNT, 2):
            return 0
        if confidence.shape != (COCO_KEYPOINT_COUNT,):
            return 0
        if not np.isfinite(keypoints).all() or not np.isfinite(confidence).all():
            return 0
        valid = (
            (confidence >= self.keypoint_conf_threshold)
            & (keypoints[:, 0] >= 0)
            & (keypoints[:, 1] >= 0)
        )
        return int(valid.sum())

    def _normalize_screen_coordinates(self, keypoints, width, height):
        normalized = keypoints.copy()
        normalized[..., 0] = normalized[..., 0] / width * 2 - 1
        normalized[..., 1] = normalized[..., 1] / width * 2 - height / width
        return normalized

    def _predict_3d(self, normalized_keypoints):
        with self.torch.no_grad():
            model_input = self.torch.from_numpy(normalized_keypoints).unsqueeze(0)
            prediction = self.model(model_input).detach().cpu().numpy()

        if prediction.ndim == 4:
            prediction = prediction[0]
        if prediction.shape[-2:] != (COCO_KEYPOINT_COUNT, 3):
            raise ValueError(
                f"3D pose model returned shape {prediction.shape}; expected (frames, 17, 3)."
            )
        return prediction

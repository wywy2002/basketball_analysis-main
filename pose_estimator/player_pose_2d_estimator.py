import numpy as np

from utils import read_stub, save_stub


COCO_KEYPOINT_COUNT = 17
POSE_2D_SCHEMA_VERSION = 3


class PlayerPose2DEstimator:
    """
    Estimates 2D body keypoints for already-tracked players.

    The pose model is run on the full frame first, then pose detections are
    matched to existing player tracks. Crop fallback is used only for unmatched
    tracks and still passes through the same quality gates.
    """

    def __init__(
        self,
        model_path,
        conf=0.25,
        padding=0.30,
        batch_size=16,
        match_threshold=0.25,
        min_valid_keypoints=8,
        enable_crop_fallback=True,
        keypoint_conf_threshold=0.20,
        max_outside_bbox_ratio=0.45,
        joint_filter_padding=0.25,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.padding = padding
        self.batch_size = batch_size
        self.match_threshold = match_threshold
        self.min_valid_keypoints = min_valid_keypoints
        self.enable_crop_fallback = enable_crop_fallback
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.max_outside_bbox_ratio = max_outside_bbox_ratio
        self.joint_filter_padding = joint_filter_padding
        self.last_stats = self._new_stats()

    def get_pose_tracks(self, frames, player_tracks, read_from_stub=False, stub_path=None):
        pose_tracks = read_stub(read_from_stub, stub_path)
        if self.is_valid_pose_cache(pose_tracks, len(frames), self.min_valid_keypoints):
            return pose_tracks

        self.last_stats = self._new_stats()
        pose_tracks = []

        for start in range(0, len(frames), self.batch_size):
            frame_batch = frames[start:start + self.batch_size]
            track_batch = player_tracks[start:start + self.batch_size]
            full_frame_results = self.model.predict(frame_batch, conf=self.conf, verbose=False)

            for frame, frame_tracks, result in zip(frame_batch, track_batch, full_frame_results):
                pose_tracks.append(self._estimate_frame_poses(frame, frame_tracks, result))

        if stub_path is not None:
            save_stub(stub_path, pose_tracks)
        return pose_tracks

    @classmethod
    def is_valid_pose_cache(cls, pose_tracks, frame_count, min_valid_keypoints=8):
        if pose_tracks is None or len(pose_tracks) != frame_count:
            return False

        for frame_poses in pose_tracks:
            if not isinstance(frame_poses, dict):
                return False
            for pose in frame_poses.values():
                if not cls.is_valid_pose_record(pose, min_valid_keypoints):
                    return False
        return True

    @classmethod
    def is_valid_pose_record(cls, pose, min_valid_keypoints=8):
        if not isinstance(pose, dict):
            return False
        if pose.get("schema_version") != POSE_2D_SCHEMA_VERSION:
            return False

        keypoints = np.asarray(pose.get("keypoints_2d"), dtype=np.float32)
        confidence = np.asarray(pose.get("confidence"), dtype=np.float32)
        if keypoints.shape != (COCO_KEYPOINT_COUNT, 2):
            return False
        if confidence.shape != (COCO_KEYPOINT_COUNT,):
            return False
        if not np.isfinite(keypoints).all() or not np.isfinite(confidence).all():
            return False

        valid_count = int(pose.get("valid_keypoints", 0))
        if valid_count < min_valid_keypoints:
            return False
        return True

    def stats_summary(self):
        stats = self.last_stats
        total_poses = stats["matched"] + stats["fallback_matched"]
        average_valid = (
            stats["valid_keypoint_sum"] / total_poses
            if total_poses > 0
            else 0.0
        )
        return (
            f"2D pose stats: tracks={stats['tracks']}, matched={stats['matched']}, "
            f"fallback={stats['fallback_matched']}, rejected={stats['rejected']}, "
            f"avg_valid_keypoints={average_valid:.2f}"
        )

    def _new_stats(self):
        return {
            "tracks": 0,
            "matched": 0,
            "fallback_matched": 0,
            "rejected": 0,
            "valid_keypoint_sum": 0,
        }

    def _estimate_frame_poses(self, frame, frame_tracks, full_frame_result):
        frame_shape = frame.shape[:2]
        candidates = self._extract_pose_candidates(full_frame_result, (0, 0), frame_shape)
        matched_tracks, used_candidates = self._match_candidates_to_tracks(
            frame_tracks,
            candidates,
            frame_shape,
        )

        if self.enable_crop_fallback:
            unmatched = [
                (track_id, player)
                for track_id, player in frame_tracks.items()
                if track_id not in matched_tracks
            ]
            fallback_tracks = self._estimate_crop_fallbacks(frame, unmatched, frame_shape)
            matched_tracks.update(fallback_tracks)

        self.last_stats["tracks"] += len(frame_tracks)
        self.last_stats["matched"] += len(used_candidates)
        self.last_stats["fallback_matched"] += sum(
            1 for pose in matched_tracks.values() if pose.get("source") == "crop"
        )
        self.last_stats["rejected"] += len(frame_tracks) - len(matched_tracks)
        self.last_stats["valid_keypoint_sum"] += sum(
            pose.get("valid_keypoints", 0) for pose in matched_tracks.values()
        )
        return matched_tracks

    def _match_candidates_to_tracks(self, frame_tracks, candidates, frame_shape):
        scored_matches = []
        for track_id, player in frame_tracks.items():
            player_bbox = player["bbox"]
            for candidate_idx, candidate in enumerate(candidates):
                if not self._candidate_passes_quality(candidate, player_bbox, frame_shape):
                    continue
                match_score = self._calculate_match_score(player_bbox, candidate)
                if match_score >= self.match_threshold:
                    scored_matches.append((match_score, track_id, candidate_idx))

        scored_matches.sort(reverse=True, key=lambda item: item[0])
        matched_tracks = {}
        used_candidates = set()
        used_tracks = set()

        for match_score, track_id, candidate_idx in scored_matches:
            if track_id in used_tracks or candidate_idx in used_candidates:
                continue
            candidate = candidates[candidate_idx]
            pose_record = self._make_pose_record(
                frame_tracks[track_id]["bbox"],
                candidate,
                "full_frame",
                match_score,
                frame_shape,
            )
            if pose_record is None:
                continue
            matched_tracks[track_id] = pose_record
            used_tracks.add(track_id)
            used_candidates.add(candidate_idx)

        return matched_tracks, used_candidates

    def _estimate_crop_fallbacks(self, frame, unmatched_tracks, frame_shape):
        fallback_tracks = {}
        crops = []
        crop_meta = []

        for track_id, player in unmatched_tracks:
            crop, offset = self._crop_player(frame, player["bbox"])
            if crop is None:
                continue
            crops.append(crop)
            crop_meta.append((track_id, player["bbox"], offset))

        for start in range(0, len(crops), self.batch_size):
            batch = crops[start:start + self.batch_size]
            batch_meta = crop_meta[start:start + self.batch_size]
            results = self.model.predict(batch, conf=self.conf, verbose=False)

            for result, (track_id, bbox, offset) in zip(results, batch_meta):
                candidates = self._extract_pose_candidates(result, offset, frame_shape)
                best_candidate = None
                best_score = -1.0
                for candidate in candidates:
                    if not self._candidate_passes_quality(candidate, bbox, frame_shape):
                        continue
                    score = self._calculate_match_score(bbox, candidate)
                    if score > best_score:
                        best_candidate = candidate
                        best_score = score

                if best_candidate is not None and best_score >= self.match_threshold:
                    pose_record = self._make_pose_record(
                        bbox,
                        best_candidate,
                        "crop",
                        best_score,
                        frame_shape,
                    )
                    if pose_record is not None:
                        fallback_tracks[track_id] = pose_record

        return fallback_tracks

    def _extract_pose_candidates(self, result, offset, frame_shape):
        if result is None or result.keypoints is None or result.keypoints.xy is None:
            return []

        keypoints_xy = self._to_numpy(result.keypoints.xy)
        if keypoints_xy.ndim != 3 or keypoints_xy.shape[1:] != (COCO_KEYPOINT_COUNT, 2):
            return []
        if len(keypoints_xy) == 0:
            return []

        if getattr(result.keypoints, "conf", None) is not None:
            keypoint_conf = self._to_numpy(result.keypoints.conf)
        else:
            keypoint_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)

        if keypoint_conf.shape != (len(keypoints_xy), COCO_KEYPOINT_COUNT):
            return []

        boxes = self._extract_boxes(result, len(keypoints_xy))
        candidates = []
        offset_x, offset_y = offset

        for idx, keypoints in enumerate(keypoints_xy):
            confidence = keypoint_conf[idx].astype(np.float32)
            full_keypoints = keypoints.astype(np.float32).copy()
            full_keypoints[:, 0] += offset_x
            full_keypoints[:, 1] += offset_y

            pose_bbox = boxes[idx] if boxes is not None else self._bbox_from_keypoints(
                full_keypoints,
                confidence,
            )
            if pose_bbox is None:
                continue
            pose_bbox = [
                float(pose_bbox[0] + offset_x if boxes is not None else pose_bbox[0]),
                float(pose_bbox[1] + offset_y if boxes is not None else pose_bbox[1]),
                float(pose_bbox[2] + offset_x if boxes is not None else pose_bbox[2]),
                float(pose_bbox[3] + offset_y if boxes is not None else pose_bbox[3]),
            ]

            valid_mask = self._valid_keypoint_mask(full_keypoints, confidence, frame_shape)
            valid_count = int(valid_mask.sum())
            pose_score = float(confidence[valid_mask].mean()) if valid_count else 0.0
            candidates.append(
                {
                    "keypoints": full_keypoints,
                    "confidence": confidence,
                    "pose_bbox": pose_bbox,
                    "pose_score": pose_score,
                    "valid_count": valid_count,
                }
            )
        return candidates

    def _extract_boxes(self, result, expected_count):
        boxes_obj = getattr(result, "boxes", None)
        if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
            return None
        boxes = self._to_numpy(boxes_obj.xyxy).astype(np.float32)
        if boxes.shape != (expected_count, 4):
            return None
        return boxes

    def _candidate_passes_quality(self, candidate, target_bbox, frame_shape):
        keypoints = candidate["keypoints"]
        confidence = candidate["confidence"]
        if keypoints.shape != (COCO_KEYPOINT_COUNT, 2):
            return False
        if confidence.shape != (COCO_KEYPOINT_COUNT,):
            return False
        if not np.isfinite(keypoints).all() or not np.isfinite(confidence).all():
            return False

        valid_mask = self._valid_keypoint_mask(keypoints, confidence, frame_shape)
        valid_count = int(valid_mask.sum())
        if valid_count < self.min_valid_keypoints:
            return False

        expanded_bbox = self._expand_bbox(target_bbox, 0.35, frame_shape)
        valid_points = keypoints[valid_mask]
        outside_mask = (
            (valid_points[:, 0] < expanded_bbox[0])
            | (valid_points[:, 0] > expanded_bbox[2])
            | (valid_points[:, 1] < expanded_bbox[1])
            | (valid_points[:, 1] > expanded_bbox[3])
        )
        outside_ratio = float(outside_mask.mean()) if len(valid_points) else 1.0
        return outside_ratio <= self.max_outside_bbox_ratio

    def _calculate_match_score(self, player_bbox, candidate):
        pose_bbox = candidate["pose_bbox"]
        iou = self._bbox_iou(player_bbox, pose_bbox)
        center_score = self._center_score(player_bbox, pose_bbox)
        keypoint_score = min(candidate["valid_count"] / COCO_KEYPOINT_COUNT, 1.0)
        pose_score = candidate["pose_score"]
        return float(
            0.45 * iou
            + 0.25 * center_score
            + 0.20 * keypoint_score
            + 0.10 * pose_score
        )

    def _make_pose_record(self, original_bbox, candidate, source, match_score, frame_shape):
        candidate = self._sanitize_candidate_for_bbox(candidate, original_bbox, frame_shape)
        if candidate is None:
            return None

        return {
            "bbox": [float(v) for v in original_bbox],
            "keypoints_2d": candidate["keypoints"].astype(float).tolist(),
            "confidence": candidate["confidence"].astype(float).tolist(),
            "source": source,
            "pose_score": float(candidate["pose_score"]),
            "match_score": float(match_score),
            "valid_keypoints": int(candidate["valid_count"]),
            "schema_version": POSE_2D_SCHEMA_VERSION,
        }

    def _sanitize_candidate_for_bbox(self, candidate, target_bbox, frame_shape):
        keypoints = candidate["keypoints"].astype(np.float32).copy()
        confidence = candidate["confidence"].astype(np.float32).copy()
        expanded_bbox = self._expand_bbox(target_bbox, self.joint_filter_padding, frame_shape)

        valid_mask = self._valid_keypoint_mask(keypoints, confidence, frame_shape)
        inside_track_mask = (
            (keypoints[:, 0] >= expanded_bbox[0])
            & (keypoints[:, 0] <= expanded_bbox[2])
            & (keypoints[:, 1] >= expanded_bbox[1])
            & (keypoints[:, 1] <= expanded_bbox[3])
        )
        keep_mask = valid_mask & inside_track_mask
        if int(keep_mask.sum()) < self.min_valid_keypoints:
            return None

        keypoints[~keep_mask] = 0.0
        confidence[~keep_mask] = 0.0
        pose_score = float(confidence[keep_mask].mean()) if int(keep_mask.sum()) else 0.0

        sanitized = dict(candidate)
        sanitized["keypoints"] = keypoints
        sanitized["confidence"] = confidence
        sanitized["valid_count"] = int(keep_mask.sum())
        sanitized["pose_score"] = pose_score
        return sanitized

    def _crop_player(self, frame, bbox):
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 1 or height <= 1:
            return None, None

        pad_x = width * self.padding
        pad_y = height * self.padding
        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(frame_w, int(x2 + pad_x))
        crop_y2 = min(frame_h, int(y2 + pad_y))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None, None
        return frame[crop_y1:crop_y2, crop_x1:crop_x2], (crop_x1, crop_y1)

    def _valid_keypoint_mask(self, keypoints, confidence, frame_shape):
        frame_h, frame_w = frame_shape
        return (
            (confidence >= self.keypoint_conf_threshold)
            & (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] < frame_w)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] < frame_h)
        )

    def _bbox_from_keypoints(self, keypoints, confidence):
        valid = confidence >= self.keypoint_conf_threshold
        if int(valid.sum()) < 2:
            return None
        valid_points = keypoints[valid]
        return [
            float(valid_points[:, 0].min()),
            float(valid_points[:, 1].min()),
            float(valid_points[:, 0].max()),
            float(valid_points[:, 1].max()),
        ]

    def _expand_bbox(self, bbox, ratio, frame_shape):
        frame_h, frame_w = frame_shape
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return [
            max(0.0, float(x1 - width * ratio)),
            max(0.0, float(y1 - height * ratio)),
            min(float(frame_w - 1), float(x2 + width * ratio)),
            min(float(frame_h - 1), float(y2 + height * ratio)),
        ]

    def _bbox_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        return float(inter_area / union) if union > 0 else 0.0

    def _center_score(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        center_a = np.array([(ax1 + ax2) / 2, (ay1 + ay2) / 2], dtype=np.float32)
        center_b = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2], dtype=np.float32)
        diagonal = np.linalg.norm(np.array([ax2 - ax1, ay2 - ay1], dtype=np.float32))
        if diagonal <= 0:
            return 0.0
        normalized_distance = np.linalg.norm(center_a - center_b) / diagonal
        return float(max(0.0, 1.0 - normalized_distance))

    def _to_numpy(self, value):
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

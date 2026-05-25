import cv2
import numpy as np
import sys

sys.path.append('../')
from utils import read_stub, save_stub


class TeamAssigner:
    """
    Assign players to two teams using jersey color.

    The assignment is intentionally done in a full-video pass. Per-frame color
    observations are noisy when players overlap or the tracker swaps identities,
    so final teams are decided by track segments instead of immediate memory.
    """

    STUB_METHOD = "color_kmeans_v3"
    UNKNOWN_TEAM = -1

    def __init__(
        self,
        sample_stride=5,
        min_crop_size=20,
        default_team=1,
        max_cluster_distance=45.0,
        min_cluster_margin=20.0,
        mixed_color_std_threshold=48.0,
        min_valid_pixel_ratio=0.20,
        min_segment_frames=4,
        min_segment_votes=3,
        min_segment_vote_ratio=0.60,
    ):
        self.sample_stride = sample_stride
        self.min_crop_size = min_crop_size
        self.default_team = default_team
        self.max_cluster_distance = max_cluster_distance
        self.min_cluster_margin = min_cluster_margin
        self.mixed_color_std_threshold = mixed_color_std_threshold
        self.min_valid_pixel_ratio = min_valid_pixel_ratio
        self.min_segment_frames = min_segment_frames
        self.min_segment_votes = min_segment_votes
        self.min_segment_vote_ratio = min_segment_vote_ratio

        self.cluster_centers = None
        self.cluster_to_team = {}
        self.last_diagnostics = {}

    def _clip_bbox(self, frame, bbox):
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]

        x1 = max(0, min(x1, frame_width - 1))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height - 1))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def _crop_region(self, frame, clipped_bbox, region):
        x1, y1, x2, y2 = clipped_bbox
        width = x2 - x1
        height = y2 - y1

        rx1, rx2, ry1, ry2 = region
        crop_x1 = x1 + int(width * rx1)
        crop_x2 = x1 + int(width * rx2)
        crop_y1 = y1 + int(height * ry1)
        crop_y2 = y1 + int(height * ry2)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        return frame[crop_y1:crop_y2, crop_x1:crop_x2]

    def _get_jersey_crop_candidates(self, frame, bbox):
        clipped = self._clip_bbox(frame, bbox)
        if clipped is None:
            return []

        x1, y1, x2, y2 = clipped
        width = x2 - x1
        height = y2 - y1
        if width < self.min_crop_size or height < self.min_crop_size:
            return []

        regions = [
            (0.25, 0.75, 0.15, 0.55),
            (0.18, 0.82, 0.18, 0.50),
            (0.30, 0.70, 0.20, 0.60),
            (0.22, 0.78, 0.12, 0.44),
            (0.28, 0.72, 0.28, 0.64),
        ]

        candidates = []
        for region in regions:
            crop = self._crop_region(frame, clipped, region)
            if crop is not None and crop.size > 0:
                candidates.append(crop)
        return candidates

    def _feature_from_crop(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        mask = (value > 45) & ((saturation > 18) | (value > 130))
        valid_pixels = lab[mask]

        crop_area = crop.shape[0] * crop.shape[1]
        min_valid_pixels = max(30, int(crop_area * self.min_valid_pixel_ratio))
        if valid_pixels.shape[0] < min_valid_pixels:
            return None

        color_std = float(np.mean(np.std(valid_pixels.astype(np.float32), axis=0)))
        if color_std > self.mixed_color_std_threshold:
            return None

        valid_ratio = float(valid_pixels.shape[0] / crop_area)
        feature = np.median(valid_pixels, axis=0).astype(np.float32)
        quality = (valid_ratio * 100.0) - color_std

        return {
            "feature": feature,
            "quality": quality,
            "valid_ratio": valid_ratio,
            "color_std": color_std,
        }

    def _extract_best_color_feature(self, frame, bbox):
        best_candidate = None
        for crop in self._get_jersey_crop_candidates(frame, bbox):
            candidate = self._feature_from_crop(crop)
            if candidate is None:
                continue
            if best_candidate is None or candidate["quality"] > best_candidate["quality"]:
                best_candidate = candidate
        return best_candidate

    def _extract_color_feature(self, frame, bbox):
        candidate = self._extract_best_color_feature(frame, bbox)
        if candidate is None:
            return None
        return candidate["feature"]

    def _collect_color_samples(self, video_frames, player_tracks):
        samples = []

        for frame_num in range(0, len(video_frames), self.sample_stride):
            frame = video_frames[frame_num]
            for track in player_tracks[frame_num].values():
                candidate = self._extract_best_color_feature(frame, track["bbox"])
                if candidate is not None:
                    samples.append(candidate["feature"])

        return samples

    def _fit_team_clusters(self, video_frames, player_tracks):
        samples = self._collect_color_samples(video_frames, player_tracks)

        if len(samples) < 2:
            self.cluster_centers = None
            self.cluster_to_team = {}
            return

        sample_array = np.array(samples, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, _, centers = cv2.kmeans(
            sample_array,
            2,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )

        center_distance = np.linalg.norm(centers[0] - centers[1])
        if center_distance < 8.0:
            self.cluster_centers = None
            self.cluster_to_team = {}
            return

        self.cluster_centers = centers.astype(np.float32)

        # LAB channel 0 is lightness. The brighter jersey cluster is Team 1.
        brighter_cluster = int(np.argmax(self.cluster_centers[:, 0]))
        darker_cluster = 1 - brighter_cluster
        self.cluster_to_team = {
            brighter_cluster: 1,
            darker_cluster: 2,
        }

    def _classify_feature_with_confidence(self, feature):
        if self.cluster_centers is None:
            return {
                "team": self.UNKNOWN_TEAM,
                "confidence": 0.0,
                "nearest_distance": None,
                "margin": None,
            }

        distances = np.linalg.norm(self.cluster_centers - feature, axis=1)
        sorted_cluster_ids = np.argsort(distances)
        cluster_id = int(sorted_cluster_ids[0])
        next_best_cluster_id = int(sorted_cluster_ids[1])
        nearest_distance = float(distances[cluster_id])
        margin = float(distances[next_best_cluster_id] - distances[cluster_id])

        if nearest_distance > self.max_cluster_distance:
            team = self.UNKNOWN_TEAM
        elif margin < self.min_cluster_margin:
            team = self.UNKNOWN_TEAM
        else:
            team = self.cluster_to_team.get(cluster_id, self.UNKNOWN_TEAM)

        confidence = max(0.0, margin) + max(0.0, self.max_cluster_distance - nearest_distance)
        return {
            "team": team,
            "confidence": float(confidence),
            "nearest_distance": nearest_distance,
            "margin": margin,
        }

    def _classify_feature(self, feature):
        return self._classify_feature_with_confidence(feature)["team"]

    def _observe_team(self, frame, player_bbox):
        best_known_observation = None
        best_unknown_observation = None

        for crop in self._get_jersey_crop_candidates(frame, player_bbox):
            candidate = self._feature_from_crop(crop)
            if candidate is None:
                continue

            classification = self._classify_feature_with_confidence(candidate["feature"])
            observation = {
                **classification,
                "feature_quality": candidate["quality"],
                "valid_ratio": candidate["valid_ratio"],
                "color_std": candidate["color_std"],
            }

            if observation["team"] == self.UNKNOWN_TEAM:
                if (
                    best_unknown_observation is None
                    or observation["feature_quality"] > best_unknown_observation["feature_quality"]
                ):
                    best_unknown_observation = observation
                continue

            if (
                best_known_observation is None
                or observation["confidence"] > best_known_observation["confidence"]
            ):
                best_known_observation = observation

        if best_known_observation is not None:
            return best_known_observation

        if best_unknown_observation is not None:
            return best_unknown_observation

        return {
            "team": self.UNKNOWN_TEAM,
            "confidence": 0.0,
            "feature_quality": 0.0,
        }

    def get_player_team(self, frame, player_bbox, player_id=None):
        observation = self._observe_team(frame, player_bbox)
        if observation["team"] == self.UNKNOWN_TEAM:
            return self.default_team
        return observation["team"]

    def _collect_raw_observations(self, video_frames, player_tracks):
        raw_observations = []
        track_observations = {}

        for frame_num, player_track in enumerate(player_tracks):
            frame_observations = {}
            frame = video_frames[frame_num]
            for player_id, track in player_track.items():
                observation = self._observe_team(frame, track["bbox"])
                frame_observations[player_id] = observation
                track_observations.setdefault(player_id, []).append(
                    {
                        "frame_num": frame_num,
                        "team": observation["team"],
                        "confidence": observation["confidence"],
                    }
                )
            raw_observations.append(frame_observations)

        return raw_observations, track_observations

    def _resolve_observation_run_team(self, observations):
        known_observations = [
            obs for obs in observations if obs["team"] != self.UNKNOWN_TEAM
        ]
        if len(known_observations) < self.min_segment_votes:
            return self.UNKNOWN_TEAM, 0.0

        team_scores = {}
        for observation in known_observations:
            team_scores[observation["team"]] = (
                team_scores.get(observation["team"], 0.0) + observation["confidence"]
            )

        best_team = max(team_scores, key=team_scores.get)
        best_score = team_scores[best_team]
        total_score = sum(team_scores.values())
        vote_ratio = best_score / total_score if total_score > 0 else 0.0

        if vote_ratio < self.min_segment_vote_ratio:
            return self.UNKNOWN_TEAM, vote_ratio

        return best_team, vote_ratio

    def _build_track_segments(self, track_observations):
        segments_by_track = {}

        for player_id, observations in track_observations.items():
            if not observations:
                segments_by_track[player_id] = []
                continue

            segments = []
            current_start_idx = 0
            current_team = self.UNKNOWN_TEAM
            candidate_team = None
            candidate_start_idx = None
            candidate_count = 0

            for idx, observation in enumerate(observations):
                observed_team = observation["team"]
                if observed_team == self.UNKNOWN_TEAM:
                    continue

                if current_team == self.UNKNOWN_TEAM:
                    initial_team, _ = self._resolve_observation_run_team(
                        observations[current_start_idx : idx + 1]
                    )
                    if initial_team != self.UNKNOWN_TEAM:
                        current_team = initial_team
                    continue

                if observed_team == current_team:
                    candidate_team = None
                    candidate_start_idx = None
                    candidate_count = 0
                    continue

                if candidate_team == observed_team:
                    candidate_count += 1
                else:
                    candidate_team = observed_team
                    candidate_start_idx = idx
                    candidate_count = 1

                if candidate_count >= self.min_segment_frames:
                    previous_end_idx = max(candidate_start_idx - 1, current_start_idx)
                    previous_observations = observations[current_start_idx : previous_end_idx + 1]
                    previous_team, previous_vote_ratio = self._resolve_observation_run_team(
                        previous_observations
                    )
                    if previous_team == self.UNKNOWN_TEAM:
                        previous_team = current_team

                    segments.append(
                        {
                            "start_frame": observations[current_start_idx]["frame_num"],
                            "end_frame": observations[previous_end_idx]["frame_num"],
                            "team": previous_team,
                            "vote_ratio": previous_vote_ratio,
                        }
                    )

                    current_start_idx = candidate_start_idx
                    current_team = observed_team
                    candidate_team = None
                    candidate_start_idx = None
                    candidate_count = 0

            final_observations = observations[current_start_idx:]
            final_team, final_vote_ratio = self._resolve_observation_run_team(final_observations)
            if final_team == self.UNKNOWN_TEAM and current_team != self.UNKNOWN_TEAM:
                final_team = current_team

            if final_team == self.UNKNOWN_TEAM:
                final_team = self._resolve_track_majority_team(observations)

            segments.append(
                {
                    "start_frame": observations[current_start_idx]["frame_num"],
                    "end_frame": observations[-1]["frame_num"],
                    "team": final_team,
                    "vote_ratio": final_vote_ratio,
                }
            )

            segments_by_track[player_id] = segments

        return segments_by_track

    def _resolve_track_majority_team(self, observations):
        known_observations = [
            obs for obs in observations if obs["team"] != self.UNKNOWN_TEAM
        ]
        if not known_observations:
            return self.default_team

        team_scores = {}
        for observation in known_observations:
            team_scores[observation["team"]] = (
                team_scores.get(observation["team"], 0.0) + observation["confidence"]
            )
        return max(team_scores, key=team_scores.get)

    def _find_segment_team(self, frame_num, segments):
        for segment in segments:
            if segment["start_frame"] <= frame_num <= segment["end_frame"]:
                return segment["team"]
        return self.default_team

    def _assign_segment_teams(self, player_tracks, segments_by_track):
        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})
            for player_id in player_track.keys():
                segments = segments_by_track.get(player_id, [])
                player_assignment[frame_num][player_id] = self._find_segment_team(
                    frame_num,
                    segments,
                )
        return player_assignment

    def _build_diagnostics(self, track_observations, segments_by_track):
        track_summary = {}
        for player_id, observations in track_observations.items():
            known_count = sum(1 for obs in observations if obs["team"] != self.UNKNOWN_TEAM)
            team_counts = {
                1: sum(1 for obs in observations if obs["team"] == 1),
                2: sum(1 for obs in observations if obs["team"] == 2),
                self.UNKNOWN_TEAM: sum(
                    1 for obs in observations if obs["team"] == self.UNKNOWN_TEAM
                ),
            }
            track_summary[player_id] = {
                "frames": len(observations),
                "known_observations": known_count,
                "team_counts": team_counts,
                "segments": segments_by_track.get(player_id, []),
            }

        return {
            "cluster_centers": (
                self.cluster_centers.tolist() if self.cluster_centers is not None else None
            ),
            "cluster_to_team": self.cluster_to_team,
            "tracks": track_summary,
        }

    def _read_cached_assignment(self, read_from_stub, stub_path, frame_count):
        cached = read_stub(read_from_stub, stub_path)
        if not isinstance(cached, dict):
            return None

        if cached.get("method") != self.STUB_METHOD:
            return None

        player_assignment = cached.get("player_assignment")
        if player_assignment is not None and len(player_assignment) == frame_count:
            self.last_diagnostics = cached.get("diagnostics", {})
            return player_assignment

        return None

    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        player_assignment = self._read_cached_assignment(read_from_stub, stub_path, len(video_frames))
        if player_assignment is not None:
            return player_assignment

        self._fit_team_clusters(video_frames, player_tracks)
        _, track_observations = self._collect_raw_observations(video_frames, player_tracks)
        segments_by_track = self._build_track_segments(track_observations)
        player_assignment = self._assign_segment_teams(player_tracks, segments_by_track)
        self.last_diagnostics = self._build_diagnostics(track_observations, segments_by_track)

        save_stub(
            stub_path,
            {
                "method": self.STUB_METHOD,
                "player_assignment": player_assignment,
                "diagnostics": self.last_diagnostics,
            },
        )

        return player_assignment

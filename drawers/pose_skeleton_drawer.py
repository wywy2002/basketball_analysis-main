import cv2


class PoseSkeletonDrawer:
    """
    Draws COCO-format 2D pose skeletons on video frames.
    """

    skeleton_edges = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (0, 1), (0, 2), (1, 3), (2, 4),
    ]

    def __init__(self, keypoint_color=(0, 255, 255), line_color=(0, 200, 0), min_confidence=0.25):
        self.keypoint_color = keypoint_color
        self.line_color = line_color
        self.min_confidence = min_confidence

    def draw(self, video_frames, pose_tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            frame_poses = pose_tracks[frame_num] if frame_num < len(pose_tracks) else {}

            for pose in frame_poses.values():
                keypoints = pose.get("keypoints_2d", [])
                confidence = pose.get("confidence", [])
                self._draw_pose(frame, keypoints, confidence)

            output_video_frames.append(frame)
        return output_video_frames

    def _draw_pose(self, frame, keypoints, confidence):
        for start_idx, end_idx in self.skeleton_edges:
            if not self._is_valid_keypoint(keypoints, confidence, start_idx):
                continue
            if not self._is_valid_keypoint(keypoints, confidence, end_idx):
                continue

            start = tuple(int(v) for v in keypoints[start_idx])
            end = tuple(int(v) for v in keypoints[end_idx])
            cv2.line(frame, start, end, self.line_color, 2)

        for idx, keypoint in enumerate(keypoints):
            if not self._is_valid_keypoint(keypoints, confidence, idx):
                continue
            point = tuple(int(v) for v in keypoint)
            cv2.circle(frame, point, 3, self.keypoint_color, cv2.FILLED)

    def _is_valid_keypoint(self, keypoints, confidence, idx):
        if idx >= len(keypoints) or idx >= len(confidence):
            return False
        x, y = keypoints[idx]
        return confidence[idx] >= self.min_confidence and x > 0 and y > 0

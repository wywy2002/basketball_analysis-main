import argparse
import os
import pickle
from collections import Counter

import cv2
import numpy as np


H36M_SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def load_pose_tracks(stub_path):
    with open(stub_path, "rb") as file:
        return pickle.load(file)


def select_track_id(pose_tracks):
    counts = Counter()
    for frame_tracks in pose_tracks:
        counts.update(frame_tracks.keys())
    if not counts:
        raise ValueError("No 3D poses found in the stub.")
    return counts.most_common(1)[0][0]


def project_points(points_3d, width, height, scale):
    points = np.asarray(points_3d, dtype=np.float32)
    points = points - points[0:1]

    # VideoPose3D outputs camera-space joints where z is the useful vertical
    # axis for visualization. Use y as depth, not screen height.
    yaw = np.deg2rad(-35.0)
    horizontal = points[:, 0] * np.cos(yaw) + points[:, 1] * np.sin(yaw)
    depth = -points[:, 0] * np.sin(yaw) + points[:, 1] * np.cos(yaw)
    vertical = points[:, 2]

    projected = np.zeros((len(points), 2), dtype=np.float32)
    projected[:, 0] = (horizontal + depth * 0.18) * scale + width * 0.5
    projected[:, 1] = (-vertical + depth * 0.08) * scale + height * 0.58
    return projected


def draw_grid(frame):
    height, width = frame.shape[:2]
    origin = np.array([width * 0.5, height * 0.78], dtype=np.float32)
    grid_color = (55, 55, 55)
    axis_x = np.array([1.0, -0.28], dtype=np.float32)
    axis_z = np.array([-0.75, -0.22], dtype=np.float32)

    for step in range(-4, 5):
        offset = step * 34
        a = origin + axis_x * -170 + axis_z * offset
        b = origin + axis_x * 170 + axis_z * offset
        c = origin + axis_z * -136 + axis_x * offset
        d = origin + axis_z * 136 + axis_x * offset
        cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), grid_color, 1)
        cv2.line(frame, tuple(c.astype(int)), tuple(d.astype(int)), grid_color, 1)

    cv2.line(frame, tuple(origin.astype(int)), tuple((origin + axis_x * 190).astype(int)), (80, 130, 255), 2)
    cv2.line(frame, tuple(origin.astype(int)), tuple((origin - np.array([0, 180])).astype(int)), (80, 220, 120), 2)
    cv2.line(frame, tuple(origin.astype(int)), tuple((origin + axis_z * 170).astype(int)), (255, 140, 80), 2)


def draw_pose(frame, keypoints_3d, track_id, frame_idx, total_frames, scale):
    height, width = frame.shape[:2]
    draw_grid(frame)
    points = project_points(keypoints_3d, width, height, scale)

    for start_idx, end_idx in H36M_SKELETON_EDGES:
        start = tuple(points[start_idx].astype(int))
        end = tuple(points[end_idx].astype(int))
        cv2.line(frame, start, end, (0, 210, 90), 4, cv2.LINE_AA)

    for point in points:
        cv2.circle(frame, tuple(point.astype(int)), 5, (0, 230, 255), cv2.FILLED, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"Track {track_id} | Frame {frame_idx + 1}/{total_frames}",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )


def make_visualization(pose_tracks, output_path, track_id=None, width=640, height=640, fps=24):
    if track_id is None:
        track_id = select_track_id(pose_tracks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    all_points = [
        np.asarray(frame_tracks[track_id]["keypoints_3d"], dtype=np.float32)
        for frame_tracks in pose_tracks
        if track_id in frame_tracks
    ]
    max_range = max(float(np.ptp(points, axis=0).max()) for points in all_points)
    scale = min(width, height) * 0.48 / max(max_range, 1e-6)

    for frame_idx, frame_tracks in enumerate(pose_tracks):
        frame = np.full((height, width, 3), 18, dtype=np.uint8)
        if track_id in frame_tracks:
            draw_pose(
                frame,
                frame_tracks[track_id]["keypoints_3d"],
                track_id,
                frame_idx,
                len(pose_tracks),
                scale,
            )
        else:
            cv2.putText(
                frame,
                f"Track {track_id} | Frame {frame_idx + 1}/{len(pose_tracks)}",
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "No 3D pose in this frame",
                (170, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)

    writer.release()
    return track_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_3d_stub", default="stubs/video_1/pose_3d_stubs.pkl")
    parser.add_argument(
        "--output_video",
        default="output_videos/3d_pose_reconstruction/video_1_single_player_3d_skeleton.avi",
    )
    parser.add_argument("--track_id", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args()


def main():
    args = parse_args()
    pose_tracks = load_pose_tracks(args.pose_3d_stub)
    track_id = make_visualization(
        pose_tracks,
        args.output_video,
        args.track_id,
        args.width,
        args.height,
        args.fps,
    )
    print(f"Saved 3D skeleton video: {args.output_video}")
    print(f"Visualized track id: {track_id}")


if __name__ == "__main__":
    main()

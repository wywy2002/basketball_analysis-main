import argparse
import os

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), ".ultralytics"))

from utils import read_stub
from ball_aquisition import BallAquisitionDetector
from configs import (
    BALL_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH,
    PLAYER_DETECTOR_PATH,
    POSE_3D_OUTPUT_DIR,
    POSE_3D_LIFTER_PATH,
    POSE_DETECTOR_PATH,
    STUBS_DEFAULT_PATH,
    YOLO_2D_OUTPUT_DIR,
)
from drawers import (
    BallTracksDrawer,
    FrameNumberDrawer,
    PoseSkeletonDrawer,
    PlayerTracksDrawer,
)
from pose_estimator import PlayerPose2DEstimator, Pose3DLifter
from team_assigner import TeamAssigner
from trackers import BallTracker, PlayerTracker
from utils import read_video, save_video


def get_default_stub_path(input_video, max_frames=None):
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    if max_frames is not None:
        video_name = f"{video_name}_first_{max_frames}_frames"
    return os.path.join(STUBS_DEFAULT_PATH, video_name)


def get_default_output_video_path(input_video, enable_pose, max_frames=None):
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    if max_frames is not None:
        video_name = f"{video_name}_first_{max_frames}_frames"

    if enable_pose:
        return os.path.join(POSE_3D_OUTPUT_DIR, f"{video_name}_3d_pose.avi")
    return os.path.join(YOLO_2D_OUTPUT_DIR, f"{video_name}_yolo_2d.avi")


def read_cached_list(stub_path, frame_count):
    cached = read_stub(True, stub_path)
    if cached is not None and len(cached) == frame_count:
        return cached
    return None


def read_pose_2d_cached_list(stub_path, frame_count, min_valid_keypoints):
    cached = read_stub(True, stub_path)
    if PlayerPose2DEstimator.is_valid_pose_cache(
        cached,
        frame_count,
        min_valid_keypoints,
    ):
        return cached
    return None


def log_step(message):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("--output_video", default=None)
    parser.add_argument("--stub_path", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--enable_pose", action="store_true")
    parser.add_argument("--pose_model", default=POSE_DETECTOR_PATH)
    parser.add_argument("--pose_lifter_model", default=POSE_3D_LIFTER_PATH)
    parser.add_argument("--disable_pose_draw", action="store_true")
    parser.add_argument("--pose_conf", type=float, default=0.25)
    parser.add_argument("--pose_match_threshold", type=float, default=0.25)
    parser.add_argument("--pose_min_valid_keypoints", type=int, default=8)
    parser.add_argument("--disable_pose_crop_fallback", action="store_true")
    args = parser.parse_args()

    if args.output_video is None:
        args.output_video = get_default_output_video_path(
            args.input_video,
            args.enable_pose,
            args.max_frames,
        )

    log_step(f"Reading video: {args.input_video}")
    video_frames = read_video(args.input_video)
    if args.max_frames is not None:
        video_frames = video_frames[:args.max_frames]

    stub_path = args.stub_path or get_default_stub_path(args.input_video, args.max_frames)
    frame_count = len(video_frames)
    log_step(f"Loaded {frame_count} frames. Cache path: {stub_path}")

    player_tracks_stub_path = os.path.join(stub_path, "player_track_stubs.pkl")
    player_tracks = read_cached_list(player_tracks_stub_path, frame_count)
    if player_tracks is None:
        log_step("Player cache miss. Loading YOLO player model...")
        player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
        player_tracks = player_tracker.get_object_tracks(
            video_frames,
            False,
            player_tracks_stub_path,
        )
    else:
        log_step("Player cache hit. Skipping YOLO player model load.")

    pose_2d_tracks = None
    pose_3d_tracks = None
    if args.enable_pose:
        pose_2d_stub_path = os.path.join(stub_path, "pose_2d_stubs.pkl")
        pose_2d_tracks = read_pose_2d_cached_list(
            pose_2d_stub_path,
            frame_count,
            args.pose_min_valid_keypoints,
        )
        pose_2d_recomputed = False
        if pose_2d_tracks is None:
            log_step("2D pose cache miss. Loading pose model...")
            pose_estimator = PlayerPose2DEstimator(
                args.pose_model,
                conf=args.pose_conf,
                match_threshold=args.pose_match_threshold,
                min_valid_keypoints=args.pose_min_valid_keypoints,
                enable_crop_fallback=not args.disable_pose_crop_fallback,
            )
            pose_2d_tracks = pose_estimator.get_pose_tracks(
                video_frames,
                player_tracks,
                False,
                pose_2d_stub_path,
            )
            pose_2d_recomputed = True
            log_step(pose_estimator.stats_summary())
        else:
            log_step("2D pose cache hit. Skipping pose model load.")

        pose_3d_stub_path = os.path.join(stub_path, "pose_3d_stubs.pkl")
        pose_3d_tracks = None if pose_2d_recomputed else read_cached_list(
            pose_3d_stub_path,
            frame_count,
        )
        if pose_3d_tracks is None:
            pose_lifter = Pose3DLifter(
                args.pose_lifter_model,
                min_valid_keypoints=args.pose_min_valid_keypoints,
            )
            if pose_lifter.is_available:
                log_step("3D pose cache miss. Running 2D-to-3D lifter...")
                pose_3d_tracks = pose_lifter.get_pose_tracks(
                    video_frames,
                    pose_2d_tracks,
                    False,
                    pose_3d_stub_path,
                )
            else:
                log_step(
                    f"3D pose lifter not found at {args.pose_lifter_model}. "
                    "Keeping 2D pose output only."
                )
                pose_3d_tracks = [{} for _ in video_frames]
        else:
            log_step("3D pose cache hit. Skipping 3D lifter model load.")

    ball_tracks_stub_path = os.path.join(stub_path, "ball_track_stubs.pkl")
    ball_tracks = read_cached_list(ball_tracks_stub_path, frame_count)
    if ball_tracks is None:
        log_step("Ball cache miss. Loading YOLO ball model...")
        ball_tracker = BallTracker(BALL_DETECTOR_PATH)
        ball_tracks = ball_tracker.get_object_tracks(
            video_frames,
            False,
            ball_tracks_stub_path,
        )
    else:
        log_step("Ball cache hit. Skipping YOLO ball model load.")
    ball_tracks = BallTracker.remove_wrong_detections(ball_tracks)
    ball_tracks = BallTracker.interpolate_ball_positions(ball_tracks)

    log_step("Assigning teams...")
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        True,
        os.path.join(stub_path, "player_assignment_color_stub.pkl"),
    )

    log_step("Detecting ball possession...")
    ball_aq = BallAquisitionDetector().detect_ball_possession(player_tracks, ball_tracks)

    log_step("Drawing output video...")
    p_drawer = PlayerTracksDrawer(team_1_color=(255, 255, 255), team_2_color=(0, 0, 255))
    output = p_drawer.draw(video_frames, player_tracks, player_assignment, ball_aq)
    if args.enable_pose and pose_2d_tracks is not None and not args.disable_pose_draw:
        output = PoseSkeletonDrawer().draw(output, pose_2d_tracks)
    output = BallTracksDrawer().draw(output, ball_tracks)
    output = FrameNumberDrawer().draw(output)

    save_video(output, args.output_video)
    log_step(f"Saved output video: {args.output_video}")


if __name__ == "__main__":
    main()

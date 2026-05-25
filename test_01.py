import cv2
from trackers import Tracker


def test_run():
    print("⏳ 正在加载追踪器模型...")
    # 注意：这里假设你在 models 文件夹里放了 yolov8x.pt
    # 如果你放的是 best.pt，请把下面这行改成 'models/best.pt'
    tracker = Tracker('models/yolov8x.pt')

    video_path = 'input_videos/test.mp4'
    print(f"👉 正在尝试读取视频: {video_path}")
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if ret:
        print("✅ 视频读取成功！准备进行单帧追踪测试...")
        # 强制只跑一帧，测试底层逻辑
        tracks = tracker.get_object_tracks([frame], read_from_stub=False)
        print("🎉 测试完美通过！追踪器底层工作正常！")
    else:
        print("❌ 视频读取失败！请检查 input_videos 文件夹里是不是真的有一个叫 test.mp4 的视频。")


if __name__ == "__main__":
    test_run()
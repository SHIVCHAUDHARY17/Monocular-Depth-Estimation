import cv2
import imageio
import numpy as np
import os


def make_gif(
    video_path: str = "outputs/overlay_video.mp4",
    output_path: str = "docs/demo.gif",
    start_sec: float = 0,
    duration_sec: float = 5,
    fps: int = 10,
    width: int = 640
):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * video_fps)
    total_frames = int(duration_sec * video_fps)
    frame_step = max(1, int(video_fps / fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    count = 0

    while count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            new_h = int(h * width / w)
            frame_resized = cv2.resize(frame_rgb, (width, new_h))
            frames.append(frame_resized)
        count += 1

    cap.release()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")
    print(f"Frames: {len(frames)}, Size: {width}x{new_h}")


if __name__ == "__main__":
    make_gif()
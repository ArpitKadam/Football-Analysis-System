import cv2 #type: ignore
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print(f"Total frames read: {len(frames)}")
    return frames


def save_video(frames, output_video_path):
    if not frames:
        print("Error: No frames to write! Check input video.")
        return

    output_dir = os.path.dirname(output_video_path)
    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved successfully: {output_video_path}")

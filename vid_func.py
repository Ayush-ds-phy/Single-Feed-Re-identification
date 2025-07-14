
from ultralytics import YOLO 
import cv2
def read_video(video_path):
    cap= cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames 
def save_video(out_vid_frames, out_vid_path):
    fourcc= cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (out_vid_frames[0].shape[1], out_vid_frames[0].shape[0])
    out= cv2.VideoWriter(out_vid_path, fourcc,24,frame_size)
    for frame in out_vid_frames: 
        out.write(frame)
    out.release()
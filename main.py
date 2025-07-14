from vid_func import read_video, save_video
from tracker import Tracker, draw_tracks_on_frames
import cv2


def main():
    video_frames = read_video('broadcast.mp4')
    tracker = Tracker('best.pt')

    # set to False first time to generate stub
    track = tracker.get_object_tracker(video_frames, read_from_stub=True, stub_path='track_stubs.pkl')

    annotated_frames = draw_tracks_on_frames(video_frames, track)

    save_video(annotated_frames, 'broadcast_out.avi')
    print("Tracking complete. Saved to 'broadcast_out.avi'")

if __name__ == '__main__':
    main()
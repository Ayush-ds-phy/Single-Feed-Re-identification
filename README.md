# Single-Feed-Re-identification


# ⚽ Single Feed Re-identification and Object Tracking using YOLO and ByteTrack

This project performs **object detection and tracking** on a single-camera football (soccer) broadcast feed. It uses a custom-trained YOLO model to detect **players, referees, ball, and goalkeepers**, and applies **ByteTrack** for object re-identification (Re-ID) and multi-object tracking.

## 📽️ Features

- 🧠 YOLOv8-based custom detection (`best.pt` not uploaded due to size constraints)
- 🔄 ByteTrack-based identity-preserving tracking
- 📦 Efficient batch processing (20 frames at a time)
- 🖼️ Frame-by-frame annotation (bounding boxes + labels)
- 🎥 Export annotated video
- 💾 Caching via `track_stubs.pkl` for fast re-runs

---

## 🚀 How It Works

### 1. **Read input video**  
Reads all frames from a video using OpenCV.

### 2. **Detect objects with YOLO**  
Detects objects in batch mode with confidence threshold `0.1`.

### 3. **Track with ByteTrack**  
Associates detections across frames using ByteTrack via `supervision`.

### 4. **Annotate frames**  
Draws color-coded bounding boxes and labels:
- 🔴 Red: Player
- 🔵 Blue: Referee
- ⚫ Black: Goalkeeper
- 🟡 Yellow: Ball

### 5. **Save video**  
Exports the annotated video as `output.avi`.

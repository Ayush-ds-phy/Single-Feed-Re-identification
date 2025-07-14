# ⚽ Smart Football Tracker — YOLOv8 + ByteTrack

Welcome to the **Single Feed Re-identification & Tracking** project!  
This tool helps you **detect and track football players, referees, goalkeepers, and the ball** from a single match video using AI.

Built with 💡YOLOv8 and 🔁ByteTrack, it labels each object and follows it across frames with consistent identities — like magic ✨, but powered by Python.

---

## 🚀 What This Project Does

- 🎥 Reads a football match video
- 🧠 Uses YOLOv8 to detect:
  - Players
  - Referees
  - Goalkeepers
  - The ball
- 🔄 Tracks these objects across all frames using ByteTrack
- 🖼️ Draws color-coded boxes and labels on each frame
- 💾 Saves the result as an annotated video (`broadcast_out.avi`)
- 🧠 Caches tracking results to avoid re-running the model

---
##how to run
- Make Sure all files are in the same directory ('best.pt' is not uploaded due to github upload constraints)
- open main.py and run the code 



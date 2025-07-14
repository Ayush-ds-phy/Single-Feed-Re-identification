from ultralytics import YOLO
import supervision as sv 
import pickle
import os
import cv2

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker= sv.ByteTrack()
        
    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
             detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)    
             detections +=detections_batch
        return detections 
        
    def get_object_tracker(self,frames, read_from_stub=False,stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks= pickle.load(f)
            return tracks 
        detections=self.detect_frames(frames)
        tracks={
            "Players":[],
            "referees":[],
            "ball":[],
            "goalkeeper":[]
        }
        for frame_num, detection in enumerate(detections):
            cls_names= detection.names
            cls_name_inv={v:k for k,v in cls_names.items()}
            print(cls_names)
            detection_supervision =sv.Detections.from_ultralytics(detection)

            #tracking object
            detection_with_tracker= self.tracker.update_with_detections(detection_supervision)
            print(detection_with_tracker)
            tracks["Players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            tracks["goalkeeper"].append({})
            for frame_detection in detection_with_tracker:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]
                if cls_id== cls_name_inv['player']:
                    tracks["Players"][frame_num][track_id]={"bbox":bbox}
                if cls_id== cls_name_inv['referee']:
                    tracks["referees"][frame_num][track_id]={"bbox":bbox}
                if cls_id== cls_name_inv['goalkeeper']:
                    tracks["goalkeeper"][frame_num][track_id]={"bbox":bbox}
            for frame_detection in detection_supervision:
                bbox= frame_detection[0].tolist()
                cls_id=frame_detection[3]

                if cls_id== cls_name_inv['ball']:
                     tracks["ball"][frame_num][1]={"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        return tracks 
           
def draw_tracks_on_frames(frames, tracks):
    color_player = (0, 0, 255)     # Red
    color_referee = (255, 0, 0)    # Blue
    color_ball = (0, 255, 255)     # Yellow
    color_goalkeeper = (0, 0, 0)   #Black 
    for frame_num, frame in enumerate(frames):
        players = tracks["Players"][frame_num]
        referees = tracks["referees"][frame_num]
        ball = tracks["ball"][frame_num]
        goalkeeper = tracks["goalkeeper"][frame_num]

        # Draw player boxes
        for track_id, info in players.items():
            x1, y1, x2, y2 = map(int, info["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2),color_player, 2)
            cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color_player , 2)

        # Draw referee boxes
        for track_id, info in referees.items():
            x1, y1, x2, y2 = map(int, info["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2),  color_referee, 2)
            cv2.putText(frame, f"Ref {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  color_referee, 2)

        # Draw ball box
        for track_id, info in ball.items():
            x1, y1, x2, y2 = map(int, info["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
            cv2.putText(frame, f"Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ball, 1)

        # Draw goalkeeper boxes
        for track_id, info in goalkeeper.items():
            x1,y1,x2,y2 =map(int,info["bbox"])
            cv2.rectangle(frame,(x1,y1),(x2,y2),color_goalkeeper,2)
            cv2.putText(frame, f"Goalkeeper {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_goalkeeper, 2)

    return frames

from ultralytics import YOLO   ##type: ignore
import supervision as sv  ##type: ignore
from utils.bbox_utils import get_centre_of_bbbox, get_bbox_width, get_foot_position
import pickle
import os
import cv2  ##type: ignore
import numpy as np ##type: ignore
import pandas as pd ##type: ignore

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_centre_of_bbbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position  

    def interpolate_ball_positions(self, balll_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in balll_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 16
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stubs=False, stubs_path=None):

        if read_from_stubs and stubs_path is not None and os.path.exists(stubs_path):
            with open(stubs_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players": [],
            "ball": [],
            "referee": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            #print("Class names: ", cls_names)

            cls_name_inv = {v:k for k, v in cls_names.items()}
            #print(f"Class name inverse: {cls_name_inv}")

            detection_supervision = sv.Detections.from_ultralytics(detection)

            ##convert goalkeeper to player
            for object_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_id] = cls_name_inv["player"]

            ## Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['ball'].append({})
            tracks['referee'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_name_inv["referee"]:
                    tracks['referee'][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_name_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox": bbox}

        if stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_centre, _ = get_centre_of_bbbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame,
                    center = (x_centre, y2),
                    axes = (int(width), int(0.35 * width)),
                    angle = 0.0,
                    startAngle = -45,
                    endAngle = 245,
                    color = color,
                    thickness = 2,
                    lineType = cv2.LINE_AA
                )
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_centre - rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        x2_rect = x_centre + rectangle_width // 2
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED
                        )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                    frame,
                    f"{track_id}",
                    (int(x1_text), int(y1_rect + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,0),
                    thickness=2
            )
        return frame

    def draw_traiangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_centre_of_bbbox(bbox)

        traingle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [traingle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [traingle_points], 0, (0, 0, 0), 2)
    
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control : {team_1*100:.2f} %", (1360, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control : {team_2*100:.2f} %", (1360, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referee'][frame_num]

            ## Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                bbox = player["bbox"]
                frame = self.draw_ellipse(frame, bbox, color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_traiangle(frame, player['bbox'], (0, 0, 255))


            # Draw Referee
            for track_id, referee in referee_dict.items():
                bbox = referee["bbox"]
                frame = self.draw_ellipse(frame, bbox, (255, 255, 0), track_id)
            
            # Draw Ball
            for track_id, ball in ball_dict.items():
                bbox = ball["bbox"]
                frame = self.draw_traiangle(frame, bbox, (0, 255, 0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)
        return output_video_frames

import pickle
import cv2
import numpy as np
from utils.bbox_utils import measure_distance, measure_xy_distance
import os 

class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

        self.lk_params = dict(
            winSize=(15,15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def get_camera_movement(self, frames, read_from_stubs=False, stubs_path=None):
        
        if read_from_stubs and stubs_path is not None and os.path.exists(stubs_path):
            with open(stubs_path, 'rb') as f:
                camera_movements = pickle.load(f)
                return camera_movements

        camera_movements = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_feature_point = new.ravel()
                old_feature_point = old.ravel()

                distance = measure_distance(new_feature_point, old_feature_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_feature_point, new_feature_point)

            if max_distance > self.minimum_distance:
                camera_movements[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            old_gray = frame_gray.copy()

        if stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(camera_movements, f)

        return camera_movements

    def adjust_positions_to_tracks(self, tracks, camera_movements_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movements_per_frame[frame_num]
                    positon_adjusted = [position[0] - camera_movement[0], position[1] - camera_movement[1]]
                    tracks[object][frame_num][track_id]['position_adjusted'] = positon_adjusted


    def draw_camera_movements(self, frames, camera_movements_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (550, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movements_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement along X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement along Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)
        
        return output_frames

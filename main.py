from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
import cv2
import numpy as np
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator

def main():
    video_frames = read_video("input_videos/input_video_2.mp4")

    tracker = Tracker('training/runs/detect/train/weights/best.pt')

    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stubs=True, 
                                       stubs_path="stubs/track_stubs_2.pkl")
    
    tracker.add_position_to_tracks(tracks)
    
    ## Camera Movement Estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stubs=True,
                                                                              stubs_path="stubs/camera_movement_stubs_2.pkl")
    
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    ## Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks["ball"])


    ## Assign team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame=video_frames[0], player_detections=tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)

            tracks['players'][frame_num][player_id]["team"] = team
            tracks['players'][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    ## Assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
        
    team_ball_control = np.array(team_ball_control)


    """
    ## save croppped image of a player
    for track_id, player in tracks['players'][10].items():
        frame = video_frames[10]
        bbox = player["bbox"]

        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        cv2.imwrite(f"cropped_images/player_{track_id}.jpg", cropped_image)
        break
    """

    ## Draw Annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    """    
    if not output_video_frames or output_video_frames[0] is None:
        print("Error: No valid frames generated after annotation.")
        return
    """

    ## Draw Camera Movement
    output_video_frames = camera_movement_estimator.draw_camera_movements(output_video_frames, camera_movement_per_frame)

    save_video(output_video_frames, "output_videos/output_video_2.avi")

if __name__ == "__main__":
    main()
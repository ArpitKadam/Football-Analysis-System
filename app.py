import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import yaml
from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
import pandas as pd

st.set_page_config(page_title="Football Analysis Tool", page_icon="⚽", layout="centered", initial_sidebar_state="expanded")

# Sidebar Navigation
st.sidebar.title("⚽ Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Video Analysis", "About", "Model Evaluation Results"])

def process_video(video_path):
    video_frames = read_video(video_path)
    tracker = Tracker('training/runs/detect/train/weights/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stubs=False)
    tracker.add_position_to_tracks(tracks)
    
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stubs=False)
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame=video_frames[0], player_detections=tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else None)
    
    team_ball_control = np.array(team_ball_control)
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movements(output_video_frames, camera_movement_per_frame)
    
    temp_avi = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
    save_video(output_video_frames, temp_avi.name)
    
    temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    mp4_path = temp_mp4.name
    
    os.system(f'ffmpeg -i "{temp_avi.name}" -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k "{mp4_path}" -y')
    
    return mp4_path

# Page: Home
if page == "Home":
    st.title("⚽ Football Analysis Tool")
    st.markdown("#### Analyze and track players, assign ball possession, and estimate camera movements in football matches.")
    st.markdown(" ")
    st.video("output_videos/output_video_1.mp4")
    st.markdown(" ")
    st.markdown("##### Football is more than just a game; it's a way of life. It teaches us discipline, teamwork, and the importance of never giving up. Every match is a test of skill, strategy, and resilience, where even the smallest moments can change everything. Success isn’t just about lifting trophies—it’s about the hours of practice, the sacrifices, and the unwavering belief in yourself and your team. A single pass can create history, a single goal can ignite dreams, and a single moment of brilliance can inspire millions. Football is about unity, where players from different backgrounds come together for a common goal. It teaches us that setbacks are temporary, but hard work and determination are permanent. The roar of the crowd, the rhythm of the ball, and the fight until the final whistle—this is what makes football legendary. Whether you win or lose, the true victory lies in giving it your all. Because in football, as in life, greatness is not given; it is earned.")
    st.markdown(" ")
    st.image("images\messi_quote_work_hard.jpg", width=550)

# Page: Upload Video
elif page == "Video Analysis":
    st.title("📤 Upload Your Football Video")

    uploaded_file = st.file_uploader("Upload a football match video", type=["mp4", "avi"])

    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())

        st.video(temp_file.name)

        if st.button("Process Video"):
            with st.spinner("Processing Video... This may take a while."):
                processed_video_path = process_video(temp_file.name)

            st.success("Processing complete! ✅")
            st.video(processed_video_path)

            with open(processed_video_path, "rb") as file:
                st.download_button("⬇️ Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

# Page: About
elif page == "About":
    st.title("ℹ️ About This App")
    st.write("This tool is designed to analyze football match videos using AI techniques. It detects players, assigns teams, tracks ball possession, and visualizes camera movement.")

    st.write("### Technologies Used:")
    st.write("- **Streamlit** for UI")
    st.write("- **OpenCV** for video processing")
    st.write("- **PyTorch** and **YoloV8** for object tracking")
    st.write("- **FFmpeg** for video conversion")

    st.write("### Features:")
    st.write("- **Player tracking** with AI-based detection")
    st.write("- **Team assignment** using color recognition")
    st.write("- **Ball possession analysis**")
    st.write("- **Camera movement estimation**")
    st.write("- **Download processed videos**")

    st.write("### Created by: **Arpit Kadam** 🚀")

    st.divider()

    st.title("About Me")

    st.write(
        """
        ##### Hi! I'm **Arpit Kadam**, an AI/ML Engineer passionate about building innovative solutions in Computer Vision, NLP, Deep Learning, Data Science, AI, Machine Learning and MLOps. I love tackling complex problems, deploying scalable AI models, and sharing my knowledge with the community. Connect with me through the links below! 🚀
        """
    )

    st.divider()
    
    st.markdown("""
    [![Personal Website](https://img.shields.io/badge/Personal-4CAF50?style=for-the-badge&logo=googlechrome&logoColor=white)](https://arpit-kadam.netlify.app/)
    [![Gmail](https://img.shields.io/badge/gmail-D14836?&style=for-the-badge&logo=gmail&logoColor=white)](mailto:arpitkadam922@gmail.com)  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?&style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arpitkadam/)  
    [![GitHub](https://img.shields.io/badge/GitHub-181717?&style=for-the-badge&logo=github&logoColor=white)](https://github.com/arpitkadam) 
    [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FB7A1E?style=for-the-badge&logo=buymeacoffee&logoColor=white)](https://buymeacoffee.com/arpitkadam) 
    [![DAGsHub](https://img.shields.io/badge/DAGsHub-231F20?style=for-the-badge&logo=dagshub&logoColor=white)](https://dagshub.com/ArpitKadam)  
    [![Dev.to](https://img.shields.io/badge/Dev.to-0A0A0A?&style=for-the-badge&logo=dev.to&logoColor=white)](https://dev.to/arpitkadam)  
    [![Instagram](https://img.shields.io/badge/Instagram-E1306C?&style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/arpit__kadam/)  
    """, unsafe_allow_html=True)

elif page == "Model Evaluation Results":

    st.title("📊 Model Evaluation Results")
    st.divider()

    st.subheader("Training Results")
    with st.expander("📄 View CSV content"):
        df = pd.read_csv("training/runs/detect/train/results.csv")
        st.dataframe(df)

    st.divider()

    yaml_path = "training/runs/detect/train/args.yaml"
    with open(yaml_path) as f:
        yaml_content = yaml.safe_load(f)

    st.subheader("Args.yaml Content")
    with st.expander("📜 View args.yaml"):
        st.code(yaml.dump(yaml_content, default_flow_style=False), language="yaml")
        ##st.json(yaml_content)

    st.divider()

    folder_path = "training/runs/detect/train"
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
    image_files.sort()

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        st.subheader(os.path.splitext(image_file)[0])
        with st.expander(f"🏞️ View {image_file}"):
            st.image(image, caption=image_file, use_container_width=True)
        st.divider()

    
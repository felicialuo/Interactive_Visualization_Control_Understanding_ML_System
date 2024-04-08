# Author: Felicia Luo
# 4/4/2024
# adapted from streamlit official docs
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import streamlit as st
# import pyvista as pv
# from stpyvista import stpyvista
import pandas as pd
import os
import datetime as dt
import time
import cv2
import random
from openai import OpenAI
import datetime

from utils import drawObjectDetection, drawCLIP, drawVCLIP, drawSKLTACT

##########  INIT  ##########
# paths to files
room_pcd_path = '../datasets/CoDe_Lab-poly/point_cloud_aligned_cropped.ply'
room_topview_path = '../datasets/CoDe_Lab-poly/topview.png'

DATASET_FOLDER = '../datasets/event_20240405_18_06_48_fps1_clip_1_0/'
DIR_ply = DATASET_FOLDER + 'pcd_ply'
DIR_color = DATASET_FOLDER + 'color'
DIR_depth = DATASET_FOLDER + 'depth'
DIR_obj_det = DATASET_FOLDER + 'object_detection_csv'

DIR_obj_3d_top = DATASET_FOLDER + 'object_3D_topview'

DIR_clip = DATASET_FOLDER + 'activity_recognition_clip'
DIR_vclip = DATASET_FOLDER + 'activity_recognition_vclip'

DIR_pose = DATASET_FOLDER + 'pose_estimation'
DIR_pose_img = DATASET_FOLDER + 'pose_img'
DIR_sklt_act_recog = DATASET_FOLDER + 'skeleton_act_recog'

DIR_temp_viewer = DATASET_FOLDER + 'temp_viewer'

start_hr, start_min = 18, 7
end_hr, end_min = 20, 50
start_time = dt.time(start_hr, start_min )
end_time = dt.time(end_hr, end_min)


# @st.cache_resource
# def stpv_readfile(dummy: str = "grid",):

#     # set up plotter
#     PLOTTER = pv.Plotter()

#     # axes
#     axes = pv.Axes(show_actor=False, actor_scale=2.0, line_width=5)
#     axes.origin = (0, 0, 0)
#     PLOTTER.add_actor(axes.actor)

#     # rs mesh of selected timeframe
#     if rgb_video and depth_video:
#         mesh_path = os.path.join(DIR_ply, os.listdir(DIR_ply)[timeframe_start])
#         mesh = pv.read(mesh_path)
#         # rotate mesh
#         mesh = mesh.rotate_y(180, point=axes.origin, inplace=False)
#         # read texture map
#         # tex = pv.read_texture('../sanct_candle/Model_0.jpg')
#         PLOTTER.add_mesh(mesh, show_edges=False, edge_color="k")
#         print('mesh added')

#     # read room pointcloud
#     pcd = pv.read(room_pcd_path)
#     PLOTTER.add_mesh(pcd, cmap='binary_r', show_edges=False, edge_color="k", 
#                      style='points', point_size=2)
    
#     PLOTTER.background_color = "white"
#     PLOTTER.view_isometric()
#     PLOTTER.window_size = [2000, 1000]
#     return PLOTTER

def construct_2d_viewer(t_start, t_end, fps, ifRGB, ifDepth, ifObjDet, ifPose, ifActRecog):
    # t_start = str(t_start).replace(":", "_")
    # t_end = str(t_end).replace(":", "_")

    # video_path = os.path.join(DIR_temp_viewer, t_start + "_" + t_end + ".mp4")
    video_path = os.path.join(DATASET_FOLDER, "temp_viewer.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'x264') #H264
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(video_path, fourcc, fps, (640*2, 480))

    # Calculate the difference in seconds
    date = datetime.date.today()  # Use any arbitrary date
    datetime_start = datetime.datetime.combine(date, t_start)
    datetime_end = datetime.datetime.combine(date, t_end)
    t_length = int(abs((datetime_end - datetime_start).total_seconds()))

    all_color_frames = sorted([f for f in os.listdir(DIR_color)])
    print('**********all_color_frames', len(all_color_frames))
    idx_start = next((i for i, f in enumerate(all_color_frames) if f.startswith(str(t_start).replace(":", "_"))), None)

    all_video_csv = sorted([f for f in os.listdir(DIR_vclip)])
    print('**********all_color_videos', len(all_video_csv))

    all_obj_3d = sorted([f for f in os.listdir(DIR_obj_3d_top)])
    print('**********all_obj_3d', len(all_obj_3d))

    all_pose_img = sorted([f for f in os.listdir(DIR_pose_img)])
    print('**********all_obj_3d', len(all_pose_img))


    for i in range(t_length):
        if ifRGB:
            if ifPose: color_frame_path = os.path.join(DIR_pose_img, all_pose_img[idx_start + i])
            else: color_frame_path = os.path.join(DIR_color, all_color_frames[idx_start + i])
            frame = cv2.imread(color_frame_path)

            if ifObjDet:
                # read obj det csv
                obj_det_csv_path = os.path.join(DIR_obj_det, all_color_frames[idx_start + i].replace("jpg", "csv"))
                drawObjectDetection(obj_det_csv_path, frame, ifDepth)

            
            if ifActRecog:
                clip_path = os.path.join(DIR_clip, all_color_frames[idx_start + i].replace("jpg", "csv"))
                drawCLIP(clip_path, frame)
                vclip_path = os.path.join(DIR_vclip, all_video_csv[(idx_start + i) // 60])
                drawVCLIP(vclip_path, frame)

                if ifPose:
                    sklt_act_recog_path = os.path.join(DIR_sklt_act_recog, all_video_csv[(idx_start + i) // 60])
                    drawSKLTACT(sklt_act_recog_path, frame)

            if ifObjDet and ifDepth: # +3D viewer
                obj_3d_path = os.path.join(DIR_obj_3d_top, all_obj_3d[idx_start + i])
                obj_3d = cv2.imread(obj_3d_path)
                # resize to same height
                if obj_3d.shape[0] != frame.shape[0]:
                    obj_3d = cv2.resize(obj_3d, (int(frame.shape[0] * obj_3d.shape[1] / obj_3d.shape[0]), frame.shape[0]))
                out_frame = np.hstack((frame, obj_3d))
            else: 
                room_topview = cv2.imread(room_topview_path)
                out_frame = np.hstack((frame, room_topview))
                

            out.write(out_frame)

    out.release()

    return video_path



########## SIDEBAR ##########
with st.sidebar:

    # user pick time
    # timeframe_start, timeframe_end = st.slider(
    #     f'**{'Select a timeframe to visualize'}**',
    #     0, num_recordings, (0,1)
    # )
    timeframe_start, timeframe_end = st.slider(
        'Select a timeframe to visualize',
        min_value=start_time, max_value=end_time,
        value=(start_time, dt.time(start_hr, start_min+10)),
        step=datetime.timedelta(minutes=1)
    )
    
    # set playback speed
    FPS = st.slider(
        'Playback speed', 1, 10
    )

    # data/ML switch buttons
    st.subheader('Camera')
    ifRGB = st.toggle('RGB Video')
    ifDepth = st.toggle('Depth Video')
    ifObjDet = st.toggle('Object Detection')
    ifPose = st.toggle('Pose Estimation')
    ifActRecog = st.toggle('Activity Recognition')

    # # take image
    # img_file_buffer = st.camera_input("Take a picture")

    # if img_file_buffer is not None:
    #     # To read image file buffer with OpenCV:
    #     bytes_data = img_file_buffer.getvalue()
    #     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    #     st.image(img_file_buffer)


    # Connect to GPT
    st.subheader('Ask Me Anything!')

    # Set OpenAI API key from Streamlit secrets
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display chat messages from history on app
        with st.container(height=700):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                    )
                response = st.write_stream(stream)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


##########  TITLE  ##########
st.title('What Happens in CoDe Lab?')

# viewer_2d, viewer_3d = st.columns(2)

######### 2D VIEWER #########
# with viewer_2d:
viewer_path = construct_2d_viewer(timeframe_start, timeframe_end, FPS, ifRGB, ifDepth, ifObjDet, ifPose, ifActRecog)
video_file = open(viewer_path, 'rb')
video_bytes = video_file.read()
# video_bytes = construct_2d_viewer(timeframe_start, timeframe_end, FPS, ifRGB, ifDepth, ifObjDet, ifPose, ifActRecog)
# print(video_bytes.shape)
if ifRGB: st.video(video_bytes)

obj_legend_path = os.path.join(DATASET_FOLDER, "label2color_legend.PNG")
obj_legend = cv2.imread(obj_legend_path)
obj_legend = cv2.cvtColor(obj_legend, cv2.COLOR_BGR2RGB)
if ifObjDet: st.image(obj_legend)


######### 3D VIEWER #########
# with viewer_3d:
#     # set up plotter
#     PLOTTER = pv.Plotter()

#     # axes
#     axes = pv.Axes(show_actor=False, actor_scale=2.0, line_width=5)
#     axes.origin = (0, 0, 0)
#     PLOTTER.add_actor(axes.actor)

#     # # rs mesh of selected timeframe
#     # if ifRGB and ifDepth:
#     #     mesh_path = os.path.join(DIR_ply, os.listdir(DIR_ply)[timeframe_start])
#     #     mesh = pv.read(mesh_path)
#     #     # rotate mesh
#     #     # mesh = mesh.rotate_y(180, point=axes.origin, inplace=False)
#     #     # read texture map
#     #     # tex = pv.read_texture('../sanct_candle/Model_0.jpg')
#     #     PLOTTER.add_mesh(mesh, show_edges=False, edge_color="k")

#     # read room pointcloud
#     pcd = pv.read(room_pcd_path)
#     PLOTTER.add_mesh(pcd, cmap='binary_r', show_edges=False, edge_color="k", 
#                         style='points', point_size=2)

#     # # fix camera pose
#     # st.session_state['cpos']=PLOTTER.show(cpos=st.session_state['cpos'], return_cpos=True)

#     PLOTTER.background_color = "white"
#     PLOTTER.view_isometric()
#     # PLOTTER.window_size = [640, 640]

#     stpyvista(PLOTTER)


########## DIARIES ##########
st.header('What were you feeling?')
diary_left, diary_right = st.columns(2)
with diary_left:
    diary_t_start = st.time_input('From', value=timeframe_start, step=60)
with diary_right:
    diary_t_end = st.time_input('to', value=timeframe_end, step=60)
# ask for diary entries
if 'diaries' not in st.session_state:
    st.session_state['diaries'] = []
# empty the text input box after enter
def diary_submit():
    st.session_state['diaries'].append((diary_t_start, diary_t_end, st.session_state.diary_store))
    st.session_state.diary_store = ''
# st.write(f'From {timeframe_start} to {timeframe_end}:')
st.text_input('Enter Your Diary:', key='diary_store', on_change=diary_submit, placeholder='I was feeling crazy!')

# show entered diaries
st.write('Your diaries:')
print(st.session_state['diaries'])
if st.session_state['diaries']:
    for (t_start, t_end, diary) in st.session_state['diaries']:
        st.write(f'From {t_start} to {t_end}, {diary}')


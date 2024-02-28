# author: Felicia Luo
# 2/21/2024
# adapted from streamlit official docs

import numpy as np
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import pandas as pd
import os
import datetime as dt
import time
import cv2
import random
from openai import OpenAI


# paths to 3d files
room_pcd_path = '../output/aligned.ply'
rs_recording_root = '../output/rs_recording/'
rs_mesh_dir = rs_recording_root + 'mesh'
rs_rgb_dir = rs_recording_root + 'color'
rs_depth_dir = rs_recording_root + 'depth'
rs_det_bbox_dir = rs_recording_root + 'det_3d_bbox'
rs_det_img_dir = rs_recording_root + 'detection'
rs_pose_dir = rs_recording_root + 'ViTPose'

# count all recordings
num_recordings = len(os.listdir(rs_mesh_dir))
print("num_recordings", num_recordings)

start_hr, start_min = 10, 30
end_hr, end_min = 20, 30
start_time = dt.time(start_hr, start_min )
end_time = dt.time(end_hr, end_min)

# store all diary entries
st.session_state['diaries'] = []
  



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
#         mesh_path = os.path.join(rs_mesh_dir, os.listdir(rs_mesh_dir)[timeframe_start])
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

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)




# sidebar
with st.sidebar:

    # user pick time
    timeframe_start, timeframe_end = st.slider(
        f'**{'Select a timeframe to visualize'}**',
        0, num_recordings, (0,1)
    )
    # time_slider = st.slider(
    #     f'**{'Select a timeframe to visualize'}**',
    #     min_value=start_time, max_value=end_time,
    #     value=(start_time, dt.time(start_hr, start_min+10)))

    # data/ML switch buttons
    st.subheader('Camera')
    rgb_video = st.toggle('RGB Video')
    depth_video = st.toggle('Depth Video')
    obj_det = st.toggle('Object Detection')
    pose_est = st.toggle('Pose Estimation')
    act_rec = st.toggle('Activity Recognition')

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


# title
st.title('Your 4D Diary')


# 3d viewer
# the whole room + mesh at timeframe
# stpyvista(stpv_readfile())

# set up plotter
PLOTTER = pv.Plotter()

# axes
axes = pv.Axes(show_actor=False, actor_scale=2.0, line_width=5)
axes.origin = (0, 0, 0)
PLOTTER.add_actor(axes.actor)

# rs mesh of selected timeframe
if rgb_video and depth_video:
    mesh_path = os.path.join(rs_mesh_dir, os.listdir(rs_mesh_dir)[timeframe_start])
    mesh = pv.read(mesh_path)
    # rotate mesh
    # mesh = mesh.rotate_y(180, point=axes.origin, inplace=False)
    # read texture map
    # tex = pv.read_texture('../sanct_candle/Model_0.jpg')
    PLOTTER.add_mesh(mesh, show_edges=False, edge_color="k")
    print('mesh added')

# read room pointcloud
pcd = pv.read(room_pcd_path)
PLOTTER.add_mesh(pcd, cmap='binary_r', show_edges=False, edge_color="k", 
                    style='points', point_size=2)

PLOTTER.background_color = "white"
PLOTTER.view_isometric()
PLOTTER.window_size = [2000, 1000]

stpyvista(PLOTTER)

# diary entries
diary_entry = st.text_input('Enter Your Diary:', placeholder='I was feeling crazy!')
st.session_state['diaries'].append((timeframe_start, diary_entry))

st.write('Your diaries:')
for (t, diary) in st.session_state['diaries']:
    st.write(f'At time {t}, {diary}')



print('timeframe_start', timeframe_start, 'timeframe_end', timeframe_end)

if rgb_video:
    rgb_img_path = os.path.join(rs_rgb_dir, os.listdir(rs_rgb_dir)[timeframe_start])
    st.image(rgb_img_path)

if depth_video:
    depth_img_path = os.path.join(rs_depth_dir, os.listdir(rs_depth_dir)[timeframe_start])
    st.image(depth_img_path, output_format="PNG")

if obj_det:
    det_img_path = os.path.join(rs_det_img_dir, os.listdir(rs_det_img_dir)[timeframe_start])
    st.image(det_img_path)

if pose_est:
    pose_img_path = os.path.join(rs_pose_dir, os.listdir(rs_pose_dir)[timeframe_start])
    st.image(pose_img_path)
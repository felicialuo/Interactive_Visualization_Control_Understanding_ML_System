import numpy as np
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import pandas as pd
import os
from datetime import time
import cv2
import random


# paths to 3d files
room_pcd_path = '/Users/felicialuo/Downloads/livingroom_pcd.ply'
rs_meshes_dir = '/Users/felicialuo/Downloads/rs_meshes'
# count all recordings
num_recordings = len(os.listdir(rs_meshes_dir))
start_hr, start_min = 10, 30
end_hr, end_min = 20, 30
start_time = time(start_hr, start_min )
end_time = time(end_hr, end_min)
  

# set up plotter
PLOTTER = pv.Plotter()

# axes
axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
axes.origin = (0, 0, 0)
PLOTTER.add_actor(axes.actor)

@st.cache_resource
def stpv_readfile(_file_path, dummy: str = "grid",):

    # read mesh
    # mesh = pv.read(room_pcd_path)
    # rotate mesh
    # mesh = mesh.rotate_y(180, point=axes.origin, inplace=False)
    # read texture map
    # tex = pv.read_texture('../sanct_candle/Model_0.jpg')
    # plotter.add_mesh(mesh, show_edges=False, edge_color="k")

    # read pointcloud
    pcd = pv.read(_file_path)
    PLOTTER.add_mesh(pcd, show_edges=False, edge_color="k")
    
    PLOTTER.background_color = "white"
    PLOTTER.view_isometric()
    # plotter.window_size = [2000, 1000]
    return PLOTTER

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
        # time.sleep(0.05)


# sidebar
with st.sidebar:

    # user pick time
    # time_slider = st.slider(
    #     f'**{'Select a timeframe to visualize'}**',
    #     0, num_recordings, (0,1)
    # )
    time_slider = st.slider(
        f'**{'Select a timeframe to visualize'}**',
        min_value=start_time, max_value=end_time,
        value=(start_time, time(start_hr, start_min+10)))

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

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})




# st.set_page_config(
#     page_title="Index",
#     page_icon="👋",
# )

# title
st.title('Your 4D Diary')

# 3d viewer
# the whole room
# stpyvista(stpv_readfile(room_pcd_path))
# rs mesh of select time
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
import openai
import datetime
import uuid

from utils import drawObjectDetection, drawCLIP, drawVCLIP, drawSKLTACT, compile_knowledge_base, reinit_assistant
print("25")
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

DIR_wholescene_img = DATASET_FOLDER + 'whole_scene_image'

start_hr, start_min = 18, 7
end_hr, end_min = 20, 48
start_time = dt.time(start_hr, start_min )
end_time = dt.time(end_hr, end_min)


if "ifRGB" not in st.session_state: st.session_state.ifRGB = False
if "ifDepth" not in st.session_state: st.session_state.ifDepth = False
if "ifObjDet" not in st.session_state: st.session_state.ifObjDet = False
if "ifPose" not in st.session_state: st.session_state.ifPose = False
if "ifActRecog" not in st.session_state: st.session_state.ifActRecog = False
if "timeframe_start" not in st.session_state: st.session_state.timeframe_start = False
if "timeframe_end" not in st.session_state: st.session_state.timeframe_end = False

def get_start_idx_length(t_start, t_end):

    # Calculate the difference in seconds
    date = datetime.date.today()  # Use any arbitrary date
    datetime_start = datetime.datetime.combine(date, t_start)
    datetime_end = datetime.datetime.combine(date, t_end)
    t_length = int(abs((datetime_end - datetime_start).total_seconds()))

    all_color_frames = sorted([f for f in os.listdir(DIR_color)])
    # print('**********all_color_frames', len(all_color_frames))
    idx_start = next((i for i, f in enumerate(all_color_frames) if f.startswith(str(t_start).replace(":", "_"))), None)

    return idx_start, t_length

def construct_2d_viewer(idx_start, t_length, fps, ifRGB, ifDepth, ifObjDet, ifPose, ifActRecog):
    video_path = os.path.join(DATASET_FOLDER, "temp_viewer.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'x264') #H264
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(video_path, fourcc, fps, (640*2, 480))
    
    all_color_frames = sorted([f for f in os.listdir(DIR_color)])
    # print('**********all_color_frames', len(all_color_frames))

    all_video_csv = sorted([f for f in os.listdir(DIR_vclip)])
    # print('**********all_color_videos', len(all_video_csv))

    all_obj_3d = sorted([f for f in os.listdir(DIR_obj_3d_top)])
    # print('**********all_obj_3d', len(all_obj_3d))

    all_pose_img = sorted([f for f in os.listdir(DIR_pose_img)])
    # print('**********all_pose_img', len(all_pose_img))


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

            if ifObjDet and ifDepth: # +3D view from top
                obj_3d_path = os.path.join(DIR_obj_3d_top, all_obj_3d[idx_start + i])
                obj_3d = cv2.imread(obj_3d_path)
                out_frame = np.hstack((frame, obj_3d))
            elif ifDepth: # 3d view from side
                wholescene_path = os.path.join(DIR_wholescene_img, all_color_frames[idx_start + i])
                wholescene_img =  cv2.imread(wholescene_path)
                out_frame = np.hstack((frame, wholescene_img))
            else: 
                room_topview = cv2.imread(room_topview_path)
                out_frame = np.hstack((frame, room_topview))

            # timestamp
            current_timeframe = all_color_frames[idx_start + i][:8].replace("_", ":")
            cv2.putText(out_frame, current_timeframe, (1230, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                

            out.write(out_frame)

    out.release()

    return video_path


########## SIDEBAR ##########
with st.sidebar:
    print("143")
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
    ifActRecog = st.toggle('Activity Recognition')
    ifPose = st.toggle('Pose Estimation')
    

    idx_start, t_length = get_start_idx_length(timeframe_start, timeframe_end)
    print("173")

    # Connect to GPT
    st.subheader('Ask Me Anything!')

    # Create an OpenAI client with your API key
    openai_client = openai.Client(api_key=st.secrets["OPENAI_API_KEY"]) 

    # check if any changes in timeframe and data selection, if so, 
    # update to st.session to current selection sand reinit assistant
    ifReinit = reinit_assistant(st.session_state, ifRGB, ifDepth, ifObjDet, 
                                ifPose, ifActRecog, timeframe_start, timeframe_end)
    print("182")
    
    # Adpated from https://medium.com/prompt-engineering/unleashing-the-power-of-openais-new-gpt-assistants-with-streamlit-83779294629f

    if "session_id" not in st.session_state or ifReinit: # Used to identify each session
        st.session_state.session_id = str(uuid.uuid4())

    if "run" not in st.session_state or ifReinit: # Stores the run state of the assistant
        st.session_state.run = {"status": None}

    if "messages" not in st.session_state or ifReinit: # Stores the messages of the assistant
        st.session_state.messages = []

    if "retry_error" not in st.session_state or ifReinit: # Used for error handling
        st.session_state.retry_error = 0
    
    if "assistant" not in st.session_state or ifReinit:
        print("199")
        openai.api_key = st.secrets["OPENAI_API_KEY"]

        # Load the previously created assistant
        st.session_state.assistant = openai.beta.assistants.retrieve("asst_RrxBX1nI1KvWcXOueMuqzKOI")

        knowledge_file_ids = []
        # Add knowledge base
        if ifObjDet and ifRGB:
            PATH_obj_det_knowledge = compile_knowledge_base(DATASET_FOLDER, 'object detection using YOLO', DIR_obj_det, idx_start, idx_start+t_length)
            obj_det_knowledge = openai_client.files.create(
                    file=open(PATH_obj_det_knowledge, "rb"), 
                    purpose="assistants"
            )
            knowledge_file_ids.append(obj_det_knowledge.id)
            print("obj_det_knowledge", obj_det_knowledge.id)

        if ifActRecog and ifRGB:
            PATH_clip_knowledge = compile_knowledge_base(DATASET_FOLDER, 'activity recognition using CLIP', DIR_clip, idx_start, idx_start+t_length)
            clip_knowledge = openai_client.files.create(
                    file=open(PATH_clip_knowledge, "rb"), 
                    purpose="assistants"
            )
            knowledge_file_ids.append(clip_knowledge.id)
            print("clip_knowledge", clip_knowledge.id)

            PATH_vclip_knowledge = compile_knowledge_base(DATASET_FOLDER, 'activity recognition using video finetuned CLIP', DIR_vclip, idx_start//60, (idx_start+t_length)//60)
            vclip_knowledge = openai_client.files.create(
                    file=open(PATH_vclip_knowledge, "rb"), 
                    purpose="assistants"
            )
            knowledge_file_ids.append(vclip_knowledge.id)
            print("vclip_knowledge", vclip_knowledge.id)

            if ifPose:
                PATH_sklt_act_recog_knowledge = compile_knowledge_base(DATASET_FOLDER, 'skeleton-based activity recognition', DIR_sklt_act_recog, idx_start//60, (idx_start+t_length)//60)
                sklt_act_recog_knowledge = openai_client.files.create(
                        file=open(PATH_sklt_act_recog_knowledge, "rb"), 
                        purpose="assistants"
                )
                knowledge_file_ids.append(sklt_act_recog_knowledge.id)
                print("sklt_act_recog_knowledge", sklt_act_recog_knowledge.id)

        print("238")

        # Create a new thread for this session
        st.session_state.thread = openai_client.beta.threads.create(
            metadata={
                'session_id': st.session_state.session_id,
            },
            messages=[
                    {
                        "role": "user",
                        "content": "Based on the provided logs, summarize the activities happened in CoDe Lab. Be concise.",
                        "file_ids": knowledge_file_ids,
                        # "file_ids": [],
                    }
            ]
        )

        # Add prev diaries to the thread
        if 'diaries' in st.session_state and st.session_state.diaries:
            for (t_start, t_end, diary) in st.session_state.diaries:
                st.session_state.messages = openai_client.beta.threads.messages.create(
                    thread_id=st.session_state.thread.id,
                    role="user",
                    content=f'From {t_start} to {t_end}, {diary}'
                )

        # Do a run to process the messages in the thread
        st.session_state.run = openai_client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id,
        )

        # # Retrieve the list of messages
        # st.session_state.messages = openai_client.beta.threads.messages.list(
        #     thread_id=st.session_state.thread.id
        # )
        # # Display messages
        # for message in reversed(st.session_state.messages.data):
        #     if message.role in ["user", "assistant"]:
        #         with st.chat_message(message.role):
        #             for content_part in message.content:
        #                 message_text = content_part.text.value
        #                 st.markdown(message_text)

        print("280")
    # If the run is completed, display the messages
    if hasattr(st.session_state.run, 'status') and st.session_state.run.status == "completed":
        # Retrieve the list of messages
        st.session_state.messages = openai_client.beta.threads.messages.list(
            thread_id=st.session_state.thread.id
        )
        # Display messages
        for message in reversed(st.session_state.messages.data):
            if message.role in ["user", "assistant"]:
                with st.chat_message(message.role):
                    for content_part in message.content:
                        message_text = content_part.text.value
                        st.markdown(message_text)

    print("295")
    if prompt := st.chat_input("How can I help you?"):
        with st.chat_message('user'):
            st.write(prompt)

        # Add message to the thread
        st.session_state.messages = openai_client.beta.threads.messages.create(
            thread_id=st.session_state.thread.id,
            role="user",
            content=prompt
        )

        # Do a run to process the messages in the thread
        st.session_state.run = openai_client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id,
        )
        if st.session_state.retry_error < 3:
            time.sleep(1) # Wait 1 second before checking run status
            st.rerun()

    print("315")
    # Check if 'run' object has 'status' attribute
    if hasattr(st.session_state.run, 'status'):
        # Handle the 'running' status
        if st.session_state.run.status == "running":
            with st.chat_message('assistant'):
                st.write("Thinking ......")
            if st.session_state.retry_error < 3:
                time.sleep(1)  # Short delay to prevent immediate rerun, adjust as needed
                st.rerun()

        # Handle the 'failed' status
        elif st.session_state.run.status == "failed":
            st.session_state.retry_error += 1
            with st.chat_message('assistant'):
                if st.session_state.retry_error < 3:
                    st.write("Run failed, retrying ......")
                    time.sleep(3)  # Longer delay before retrying
                    st.rerun()
                else:
                    st.error("FAILED: The OpenAI API is currently processing too many requests. Please try again later ......")

        # Handle any status that is not 'completed'
        elif st.session_state.run.status != "completed":
            # Attempt to retrieve the run again, possibly redundant if there's no other status but 'running' or 'failed'
            st.session_state.run = openai_client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=st.session_state.run.id,
            )
            if st.session_state.retry_error < 3:
                time.sleep(3)
                st.rerun()

print("348")
##########  TITLE  ##########
st.title('What Happens in CoDe Lab?')


######### 2D VIEWER #########
viewer_path = construct_2d_viewer(idx_start, t_length, FPS, ifRGB, ifDepth, ifObjDet, ifPose, ifActRecog)
video_file = open(viewer_path, 'rb')
video_bytes = video_file.read()

if ifRGB: 
    st.header("The machine's understanding:")
    st.video(video_bytes)
print("358")

obj_legend_path = os.path.join(DATASET_FOLDER, "label2color_legend.PNG")
obj_legend = cv2.imread(obj_legend_path)
obj_legend = cv2.cvtColor(obj_legend, cv2.COLOR_BGR2RGB)
if ifObjDet: st.image(obj_legend)

print("365")
########## DIARIES ##########
st.header('Would you like to share your thoughts?')
diary_left, diary_right = st.columns(2)
with diary_left:
    diary_t_start = st.time_input('From', value=timeframe_start, step=60)
with diary_right:
    diary_t_end = st.time_input('to', value=timeframe_end, step=60)
# ask for diary entries
if 'diaries' not in st.session_state:
    st.session_state.diaries = []
# empty the text input box after enter
def diary_submit():
    st.session_state.diaries.append((diary_t_start, diary_t_end, st.session_state.diary_store))
    
    # Add diary to the thread
    st.session_state.messages = openai_client.beta.threads.messages.create(
        thread_id=st.session_state.thread.id,
        role="user",
        content=f'From {diary_t_start} to {diary_t_end}, {st.session_state.diary_store}'
    )
    # Do a run to process the messages in the thread
    st.session_state.run = openai_client.beta.threads.runs.create(
        thread_id=st.session_state.thread.id,
        assistant_id=st.session_state.assistant.id,
    )

    st.session_state.diary_store = ''

current_diary = st.text_input('Enter Your Diary:', key='diary_store', on_change=diary_submit, placeholder='I was feeling crazy!')

# show entered diaries
st.write('Your diaries:')
print("All diaries so far", st.session_state.diaries)
if st.session_state.diaries:
    for (t_start, t_end, diary) in st.session_state.diaries:
        st.write(f'From {t_start} to {t_end}, {diary}')


import numpy as np
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import pandas as pd

@st.cache_resource
def stpv_readfile(dummy: str = "grid"):

    plotter = pv.Plotter()

    # axes
    axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
    axes.origin = (0, 0, 0)
    plotter.add_actor(axes.actor)

    # read mesh
    mesh = pv.read('../st_nicholas/mesh_1st.obj')
    # rotate mesh
    # mesh = mesh.rotate_y(180, point=axes.origin, inplace=False)
    # read texture map
    # tex = pv.read_texture('../sanct_candle/Model_0.jpg')
    plotter.add_mesh(mesh, show_edges=False, edge_color="k")

    # # # read pointcloud
    # # pcd = pv.read('../output/office_polycam_pcd.ply')
    # # plotter.add_mesh(pcd, show_edges=False, edge_color="k")
    
    plotter.background_color = "white"
    plotter.view_isometric()
    # plotter.window_size = [2000, 1000]
    return plotter


st.set_page_config(
    page_title="Index",
    page_icon="👋",
)

# title
st.write('# St. Nicholas Chapel, Beaver, PA')

# 3d viewer
stpyvista(stpv_readfile())

# sidebar
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

# text input
st.text_input("Comments", key="comments")

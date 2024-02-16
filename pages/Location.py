import streamlit as st
import time
import numpy as np
import pandas as pd


st.set_page_config(page_title="Location", page_icon="📈")

st.markdown("# Location")
st.sidebar.header("Loacation")

# map
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [40.69, -80.30],
    columns=['lat', 'lon'])
st.map(map_data)
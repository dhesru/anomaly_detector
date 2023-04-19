import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

import anomaly_detect
import data_visualization
import helpers
import info



EXAMPLE_NO = 1
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Anomaly Detector",  # required
                options=["Upload Data", "Detect Anomalies","Visualization of Results","Information"],
                icons=["cloud-upload", "bar-chart-fill", "robot", "clipboard-check"],  # optional
                menu_icon="tools",  # optional
                default_index=0,  # optional
            )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)


if selected == "Upload Data":
    helpers.data_uploader()
if selected == 'Detect Anomalies':
    anomaly_detect.detect_anomalies()
if selected == 'Visualization of Results':
    data_visualization.viz_anomaly_data()
if selected == 'Information':
    helpers.info()

def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

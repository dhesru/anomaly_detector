import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import anomaly_detect
import data_visualization
import helpers
import inference

st.sidebar.image("https://s3.ap-southeast-1.amazonaws.com/files-scs-prod/public%2Fimages%2F1667201979950-image.png", use_column_width=True)
EXAMPLE_NO = 1
def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                    menu_title="Anomaly Detector",  # required
                    options=["Upload Data", "Detect Anomalies","Visualization of Results","Information"],
                    icons=["cloud-upload", "bi bi-search",'robot', "bi bi-graph-up-arrow", "bi bi-info-square"],  # optional
                    menu_icon="tools",  # optional
                    default_index=0,  # optional

                    styles = {"nav-link-selected": {"background-color": "#009999"},
                         }
            )

        return selected

selected = streamlit_menu(example=EXAMPLE_NO)


if selected == "Upload Data":
    helpers.data_uploader()
if selected == 'Detect Anomalies':
    anomaly_detect.detect_anomalies()
# if selected == "Model Inference":
#     inference.main()
if selected == 'Visualization of Results':
    data_visualization.viz_anomaly_data()
if selected == 'Information':
    helpers.info()

def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

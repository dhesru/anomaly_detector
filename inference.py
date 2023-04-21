import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def infer():
    '''Code for Model Inference'''
    fitted = st.session_state.model

def main():
    st.title('Infer Based on Trained models')
    if 'model' not in st.session_state:
        st.write('Please train your model in the Detect Anomalies section.')
    else:
        infer()

import streamlit as st
import pandas as pd

def data_uploader():
    '''This function is used for uploading CSV file'''
    st.title('Welcome to Anomaly Detector')

    uploaded_file = st.file_uploader("Upload a CSV file to detect anomalies..")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        col_names = list(df.columns)
        sensors = st.multiselect(
            'Please select the sensor variables',
            col_names)
        st.session_state.df = df
        if st.button('Confirm variables selected'):
            st.session_state.sensors = sensors
            st.warning('You have selecter the following columns: ', sensors)

def info():
    '''Prints information of the page.'''
    st.title('Information')
    st.write('This anomaly detector derives condition indicators such as Standard Deviation, Mean, Kurtosis, Variance and Max using a rolling window of size 5. '
             'The derived features are then condensed into 5 components using Principal Component Analysis.'
             ' These features are the fed into Isolation forest for anomalous points to be predicted.'
             ' To begin detecting anomalous points in your multivariate time series, proceed to upload.'
             )

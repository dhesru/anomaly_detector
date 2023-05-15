import streamlit as st
import pandas as pd


def convert(string):
    list1 = []
    list1[:0] = string
    return list1

def data_uploader():
    '''This function is used for uploading CSV file'''

    uploaded_file = st.file_uploader("Upload a CSV file for the AnoBot to detect anomalies..")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file,index_col=0)
        st.dataframe(df)
        col_names = list(df.columns)
        col_names_ss = list(df.columns)
        sensors = st.multiselect(
            'Please select the sensor variables',
            col_names)
        col_names_ss.insert(0, 'No Label')

        label = st.selectbox("Please select the label column, if any. This label column shall be used for the evaluation of the model. If there are no labels is this dataset, kindly select 'No labels' option.",col_names_ss)
        st.session_state.df = df
        window_size_options = [x for x in range(5,20,5)]
        window_size = st.selectbox(
            'Select the window size you would want for computation of rolling values',window_size_options)
        if st.button('Confirm variables selected'):
            st.session_state.sensors = sensors
            st.session_state.label = label
            if window_size == None:
                window_size = 5
            st.session_state.window_size = window_size
            st.write(f"You have selected the following columns as sensor variables:  **{', '.join(sensors)}**")
            if label == 'No Label':
                st.write('You have not selected any labels, hence model evaluation metrics may not be available.')
            else:
                st.write(f"Your label column is: **{label}**")
            st.write(f"You have selected a window size of **{window_size}** for computation of rolling values.")
            st.success('You have successfully selected the required variables. Have fun detecting those anomalies!')

def info():
    '''Prints information of the page.'''
    st.title('Information')
    st.write('This anomaly detector derives condition indicators such as Standard Deviation, Mean, Kurtosis, Variance and Max using a rolling window of size 5. '
             'The derived features are then condensed into 5 components using Principal Component Analysis.'
             ' These features are the fed into Isolation forest for anomalous points to be predicted.'
             ' To begin detecting anomalous points in your multivariate time series, proceed to upload.'
             )
    st.subheader("Upload Data")
    st.write('This tab is used for uploading of Sensor readings for anomaly detection. Once you have uploaded the CSV, select the sensor readings that needs to be utilized.')
    st.subheader("Train & Detect Anomalies")
    st.write('This tab is used to detect anomalies. Once anomalies are detected, the condition indicators and evaluation metrics will be displayed.')
    st.subheader("Visualization of Results")
    st.write('This tab is used for visualization of the results. It displays the Principal Compoenents on the data and anomalous points on time series data for both training and inferenced data.')

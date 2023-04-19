import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def viz_anomaly_data():
    '''Vizualize Anomaly Detection results'''
    st.title('Visualize Detected Anomalies')
    if 'iso_results' not in st.session_state:
        st.write('Please upload time series data and run a anomaly detection model to visualize the anomalous points.')
    else:
        sensor_cols = st.session_state.sensors
        ncol = len(sensor_cols)
        results = st.session_state.iso_results
        results['anomaly_con'] = results.anomaly.apply(lambda x: 1 if x == -1 else np.nan)
        for sensor in sensor_cols:
            sensor_name = sensor + '_a'
            results[sensor_name] = results[sensor] * results.anomaly_con

        with st.expander('View Principal Component Analysis Chart'):
            if 'pca' in st.session_state:
                pca = st.session_state.pca
                df = st.session_state.df
                df['color'] = df.label.apply(lambda x: '#FF2400' if x == 1 else '#03AC13')
                plt.scatter(pca[:, 0], pca[:, 1],c=df.color,alpha=0.5,s=3)
                st.pyplot(plt)
                plt.cla()

        with st.expander("View Anomalous points on time series"):
            for i in range(ncol):
                plt.plot(results[sensor_cols[i]],color='#1520A6',alpha=0.8)
                plt.title("Anomalous points for " + str(sensor_cols[i]))
                anomaly_col = sensor_cols[i] + '_a'
                plt.scatter(y=results[anomaly_col], x=results.index, c='#FF2400',marker='o',s=18)
                st.pyplot(plt)
                plt.cla()

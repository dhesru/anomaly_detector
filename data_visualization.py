import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

models = ['iso_results','lof_results','sos_results']

def check_mod_res():
    """This function checks if trained model exists in the current session"""
    for mod in models:
        if mod in st.session_state:
            return True,mod
    return False,None


def viz_anomaly_data():
    '''Vizualize Anomaly Detection results'''
    st.title('Visualize Detected Anomalies')
    mod_avail,model = check_mod_res()
    if not mod_avail:
        st.write('Please upload time series data and run a anomaly detection model to visualize the anomalous points.')
    else:
        sensor_cols = st.session_state.sensors
        ncol = len(sensor_cols)
        results = st.session_state.get(model)
        results['anomaly_con'] = results.anomaly.apply(lambda x: 1 if x == -1 else np.nan)
        for sensor in sensor_cols:
            sensor_name = sensor + '_a'
            results[sensor_name] = results[sensor] * results.anomaly_con
        if 'pca_train' in st.session_state:
            with st.expander('View Principal Component Analysis Chart for training data'):
                pca = st.session_state.pca_train
                df = st.session_state.df_viz
                df['color'] = df.label.apply(lambda x: '#FF2400' if x == 1 else '#03AC13')
                if not isinstance(pca, np.ndarray):
                    pca = pca.to_numpy()

                plt.scatter(pca[:, 0], pca[:, 1],c=df.color,alpha=0.5,s=3)
                plt.legend()
                st.pyplot(plt)
                plt.cla()
        if 'pca_test' in st.session_state:
            with st.expander('View Principal Component Analysis Chart for inferenced data'):
                pca = st.session_state.pca_test
                df = st.session_state.df_viz_test
                if not isinstance(pca, np.ndarray):
                    pca = pca.to_numpy()
                df['color'] = df.label.apply(lambda x: '#FF2400' if x == 1 else '#03AC13')
                plt.scatter(pca[:, 0], pca[:, 1],c=df.color,alpha=0.5,s=3)
                st.pyplot(plt)
                plt.cla()

        with st.expander("View Anomalous points on time series for training data"):
            for i in range(ncol):
                plt.plot(results[sensor_cols[i]],color='#1520A6',alpha=0.8)
                plt.title("Anomalous points for " + str(sensor_cols[i]))
                anomaly_col = sensor_cols[i] + '_a'
                plt.scatter(y=results[anomaly_col], x=results.index, c='#FF2400',marker='o',s=18)
                st.pyplot(plt)
                plt.cla()
        if 'df_test' in st.session_state:
            with st.expander("View Anomalous points on time series for inferenced data"):
                sensor_cols_test = st.session_state.sensor_test
                results_test = st.session_state.iso_results_test
                ncol_t = len(sensor_cols_test)
                for i in range(ncol_t):
                    plt.plot(results_test[sensor_cols_test[i]],color='#1520A6',alpha=0.8)
                    plt.title("Anomalous points for " + str(sensor_cols_test[i]))
                    anomaly_col = sensor_cols_test[i] + '_a'
                    plt.scatter(y=results_test[anomaly_col], x=results_test.index, c='#FF2400',marker='o',s=18)
                    st.pyplot(plt)
                    plt.cla()

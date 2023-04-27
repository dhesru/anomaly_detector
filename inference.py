import streamlit as st
import pandas as pd
import numpy as np
from anomaly_detect import anomaly_detection_pipeline_iso_fst,check_label
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

model_types = {'iso':'Isolation Forest'}
pca = True
scale = True
n_comp = 2  # number of PCA components
fe = True
window = 5


def infer():
    '''Code for Model Inference'''
    fitted = st.session_state.model
    model_type = st.session_state.model_type
    model_option = np.array([model_types.get(model_type)])
    print(model_option)

    option = st.selectbox('Select the Model to infer', (model_option))
    uploaded_file = st.file_uploader("Upload CSV file to infer")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file,index_col=0)
        st.session_state.df_test = df
        col_names = list(df.columns)
        sensor_cols = st.multiselect(
            'Please select the sensor variables',
            col_names)
        st.session_state.sensor_test = sensor_cols
        col_names.insert(0, 'No Label')

        label = st.selectbox("Please select the label column, if any. This label column shall be used for the evaluation of the model. If there are no labels is this dataset, kindly select 'No labels' option.",col_names)
        if st.button('Begin Inference'):
            p, r, f1, tp, fp, fn, results, eng_fe = anomaly_detection_pipeline_iso_fst(df, scale=scale, pca=pca,
                                                                                       n_comp=n_comp,
                                                                                       fe=fe, window=window,
                                                                                       sensor_cols=sensor_cols,
                                                                                       inf=True)
            row_dict = {'precision': [p], 'recall': [r], 'f1_score': [f1]}
            df_res = pd.DataFrame.from_dict(row_dict)
            results['anomaly_con'] = results.anomaly.apply(lambda x: 1 if x == -1 else np.nan)
            st.session_state.iso_results_test = results
            st.title('Results')
            if check_label()[0]:
                st.dataframe(df_res.style.format("{:.2}"))
            else:
                st.write('Evaluation is not supported currently as no labels were provided.')

def main():
    st.title('Inference Based on Trained models')
    if 'model' not in st.session_state:
        st.write('Please train your model in the Detect Anomalies section.')
    else:
        infer()

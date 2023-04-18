import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


st.title('Anomaly Detector for Multivariate timeseries')

uploaded_file = st.file_uploader("Choose a CSV file for anomaly detection", accept_multiple_files=False)


def feature_engineer(data, window):
    '''Compute features likes Kurtosis, mean, standard deviation max, & variance'''
    df_rolling_kurt = data.rolling(window).kurt().fillna(method='bfill').fillna(method='ffill')
    df_rolling_kurt = df_rolling_kurt.rename(
        columns=dict(zip([x for x in df_rolling_kurt.columns], [x + '_kurt' for x in df_rolling_kurt.columns])))

    df_rolling_mean = data.rolling(window).mean().fillna(method='bfill').fillna(method='ffill')
    df_rolling_mean = df_rolling_mean.rename(
        columns=dict(zip([x for x in df_rolling_mean.columns], [x + '_mean' for x in df_rolling_mean.columns])))

    df_rolling_std = data.rolling(window).std().fillna(method='bfill').fillna(method='ffill')
    df_rolling_std = df_rolling_std.rename(
        columns=dict(zip([x for x in df_rolling_std.columns], [x + '_std' for x in df_rolling_std.columns])))

    df_rolling_max = data.rolling(window).max().fillna(method='bfill').fillna(method='ffill')
    df_rolling_max = df_rolling_max.rename(
        columns=dict(zip([x for x in df_rolling_max.columns], [x + '_max' for x in df_rolling_max.columns])))

    df_rolling_var = data.rolling(window).var().fillna(method='bfill').fillna(method='ffill')
    df_rolling_var = df_rolling_var.rename(
        columns=dict(zip([x for x in df_rolling_var.columns], [x + '_var' for x in df_rolling_var.columns])))
    data = pd.concat([df_rolling_kurt, df_rolling_mean, df_rolling_std, df_rolling_std,df_rolling_max,df_rolling_var], axis=1)

    return data


def compare_anomaly(anomaly, label):
    if anomaly == -1 and label == 1:
        return 'TP'
    elif anomaly == -1 and label == 0:
        return 'FP'
    elif anomaly == 1 and label == 1:
        return 'FN'
    elif anomaly == -1 and label == 0:
        return 'TN'
    else:
        return None


def get_conf_matrix_metrics(df_c):
    val_counts = df_c.hits.value_counts()
    tp = val_counts.get('TP', 0)
    fp = val_counts.get('FP', 0)
    fn = val_counts.get('FN', 0)
    return tp, fp, fn


def anomaly_detection_pipeline(df, scale, pca, n_comp, fe, window):
    df_c = df.copy()
    X = df_c[df_c.columns[df_c.columns.str.contains('condition')]]
    if fe:
        X = feature_engineer(X, window=window)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if pca:
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        X = pca.transform(X)
        X = pd.DataFrame(np.column_stack([df[['label']], X]))
        col_names = ['label']
        X_feats = list()
        for i in range(n_comp):
            col_name = 'pca_' + str(i)
            col_names.append(col_name)
            X_feats.append(col_name)
        X.columns = col_names
        X = X[X_feats]
    if isinstance(X, np.ndarray):
        isf = IsolationForest(random_state=0, contamination=0.0009).fit(X)
        df_c['anomaly'] = pd.Series(isf.predict(X))
    else:
        isf = IsolationForest(random_state=0, contamination=0.0009).fit(X.values)
        df_c['anomaly'] = pd.Series(isf.predict(X.values))
    df_c['hits'] = df_c.apply(lambda x: compare_anomaly(x.anomaly, x.label), axis=1)

    tp, fp, fn = get_conf_matrix_metrics(df_c)
    try:
        precision = (tp / (tp + fp)) * 100
    except ZeroDivisionError:
        precision = np.nan

    try:
        recall = (tp / (tp + fn)) * 100
    except ZeroDivisionError:
        recall = np.nan

    f1_score = (precision * recall) / (precision + recall)

    return precision, recall, f1_score, tp, fp, fn, df_c

def detect_anomalies(scale,pca,n_comp,fe):
    with st.spinner('Running anomaly detection. Please hold as this may take some time...'):
        print('start anomaly detection')
        p, r, f1, tp, fp, fn, results = anomaly_detection_pipeline(df, scale=scale, pca=pca, n_comp=n_comp, fe=fe,
                                                                   window=window)
        row_dict = {'precision': p, 'recall': r, 'f1_score': f1,
                     'scale': scale, 'pca': pca, 'feature_eng': fe, 'window': window, 'tp': tp, 'fp': fp, 'fn': fn}
        print('anomaly detection completed')

    results['anomaly_con'] = results.anomaly.apply(lambda x: 1 if x == -1 else np.nan)
    condition_cols = results.columns[results.columns.str.contains('condition')]

    for cond in condition_cols:
        cond_name = cond + '_a'
        results[cond_name] = results[cond] * results.anomaly_con

    st.success('Anomalies have been detected!')
    to_print = 'The red points are anomalies.'
    st.write(
        to_print)
    ncol = len(condition_cols)

    with st.expander("See Anomalous points"):
        for i in range(ncol):
            plt.plot(results[condition_cols[i]])
            plt.title("Anomalous points for " + str(condition_cols[i]))
            anomaly_col = condition_cols[i] + '_a'
            plt.scatter(y=results[anomaly_col], x=results.index, c='red')
            st.pyplot(plt)
            plt.clf()
    st.write(row_dict)

st.write('This anomaly detector derives condition indicators such as Standard Deviation, Mean,Kurtosis, Variance and Max were derived using a rolling window of size 5. '
         'The derived features are then condensed into 5 components using Principal Component Analysis'
         'These features are the fed into Isolation forest for anomalous points to be predicted'
         'To begin detecting anomalous points in your multivariate time series, proceed to upload.'
         )

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)
    pca = True
    scale= True
    n_comp = 5 # number of PCA components
    fe = True
    window = 5
    if st.button('Start Anomaly Detection'):
        detect_anomalies(scale,pca,n_comp,fe)




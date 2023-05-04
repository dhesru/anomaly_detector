import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import make_scorer,precision_score,recall_score,f1_score as f1_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from pycaret.anomaly import setup,create_model,predict_model


CONTAMINATION_RATE = 0.1

st.title('Anomaly Detector for Multivariate timeseries')


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
    data = pd.concat([df_rolling_kurt, df_rolling_mean, df_rolling_std,df_rolling_max,df_rolling_var], axis=1)

    return data

def check_label():
    label = st.session_state.label
    if label == "No Label":
        return False , label
    else:
        return True, label
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

def norm_score(x):
    '''Normalize Probability Score'''
    return 1 - (x - np.min(x)) / (np.max(x) - np.min(x))

def optmized_model(X_train,y_train,contamination_rate):
    '''Use GridSearch to optimze the model for f1_score. Obtain the best parameters and return it'''
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.values
    n_estimators = [50, 100]
    max_features = [1.0, 5, 10]
    bootstrap = [True]
    param_grid = dict(n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap)

    # Build the gridsearch
    model_isf = IsolationForest(n_estimators=n_estimators,
                                max_features=max_features,
                                contamination=contamination_rate,
                                bootstrap=False,
                                n_jobs=-1)

    # Define an f1_scorer
    f1sc = make_scorer(f1_scorer, average='macro')
    y_tr = np.where(y_train == 1, -1, 1)
    grid = GridSearchCV(estimator=model_isf, param_grid=param_grid, cv=3, scoring=f1sc)
    grid_results = grid.fit(X=X_train, y=y_tr)
    return grid_results

def fe_scale_pca(X,window,n_comp,inf,use_label,df_c,label):
    """Common utility function to perform feature engineering, scaling and PCA to condense features"""
    X = feature_engineer(X, window=window)
    st.session_state.fe_df = X

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = PCA(n_components=n_comp)
    pca.fit(X)
    X = pca.transform(X)
    # if inf:
    #     st.session_state.pca_test = X
    # else:
    #     st.session_state.pca_train = X

    if use_label:
        X = pd.DataFrame(np.column_stack([df_c[[label]], X]))
        col_names = [label]
    else:
        X = pd.DataFrame(X)
        col_names = list()
    X_feats = list()
    for i in range(n_comp):
        col_name = 'pca_' + str(i)
        col_names.append(col_name)
        X_feats.append(col_name)
    X.columns = col_names
    X = X[X_feats]

    return X



def anomaly_detection_pipeline_iso_fst(df, n_comp, window,sensor_cols,inf):
    '''Anomaly Detection pipeline using Isolation Forest'''
    df_c = df.copy()
    X = df_c[list(sensor_cols)]
    st.session_state.df_viz = df_c
    eng_fe = pd.DataFrame()
    use_label, label = check_label()
    precision, recall, f1_score, tp, fp, fn = ['No Labels'] * 6

    X = fe_scale_pca(X,window,n_comp,inf,use_label,df_c,label)

    if inf:
        if use_label:
            y = df_c[[label]]
        if isinstance(X, np.ndarray):
            isf = st.session_state.model
            df_c['anomaly'] = pd.Series(isf.predict(X))
            df_c['probability_score'] = pd.Series(norm_score(isf.decision_function(X)))
        else:
            isf = st.session_state.model
            df_c['anomaly'] = pd.Series(isf.predict(X.values))
            df_c['probability_score'] = pd.Series(norm_score(isf.decision_function(X.values)))

    else:
        if use_label:
            y = df_c[[label]]
            contamination_rate = y.value_counts().get(1) / y.value_counts().get(0)
            grid_results = optmized_model(X,y,contamination_rate)
            isf = IsolationForest(n_estimators=grid_results.best_params_.get('n_estimators'),
                                  max_features=grid_results.best_params_.get('max_features'),
                                  contamination=contamination_rate,
                                  bootstrap=grid_results.best_params_.get('bootstrap'),
                                  n_jobs=-1)
            if isinstance(X, np.ndarray):
                isf = isf.fit(X)
                df_c['anomaly'] = pd.Series(isf.predict(X))
                df_c['probability_score'] = pd.Series(norm_score(isf.decision_function(X)))
            else:
                isf = isf.fit(X.values)
                df_c['anomaly'] = pd.Series(isf.predict(X.values))
                df_c['probability_score'] = pd.Series(norm_score(isf.decision_function(X.values)))
            st.session_state.model = isf
            st.session_state.model_type = 'iso'
        else:
            if isinstance(X, np.ndarray):
                isf = IsolationForest(random_state=0, contamination=0.0009).fit(X)
                df_c['anomaly'] = pd.Series(isf.predict(X))
                df_c['probability_score'] = pd.Series(norm_score(isf.decision_function(X)))
            else:
                isf = IsolationForest(random_state=0, contamination=0.0009).fit(X.values)
                df_c['anomaly'] = pd.Series(isf.predict(X.values))
                df_c['probability_score'] = pd.Series(norm_score(isf.decision_function(X.values)))
            st.session_state.model = isf
            st.session_state.model_type = 'iso'



    if use_label:
        df_c['hits'] = df_c.apply(lambda x: compare_anomaly(x.anomaly, x[label]), axis=1)
        y_tr = np.where(y == 1, -1, 1)
        y_pred = df_c['anomaly'].to_numpy()

        tp, fp, fn = get_conf_matrix_metrics(df_c)
        recall = recall_score(y_tr, y_pred, average='macro')
        precision = precision_score(y_tr, y_pred, average='macro')
        f1_sc = f1_scorer(y_tr, y_pred, average='macro')

    return precision, recall, f1_sc, tp, fp, fn, df_c,eng_fe

def anomaly_detection_pipeline_dbscan(df, n_comp,window,sensor_cols,inf):
    '''Implementation DBSCAN algorithm for anomaly detection'''
    df_c = df.copy()
    X = df_c[list(sensor_cols)]
    st.session_state.df_viz = df_c
    eng_fe = pd.DataFrame()
    use_label, label = check_label()
    precision, recall, f1_score, tp, fp, fn = ['No Labels'] * 6
    p, r, f1, tp, fp, fn, results = [None] *8

    X = fe_scale_pca(X,window,n_comp,inf,use_label,df_c,label)
    X = X.to_numpy()

    if inf:
        if use_label:
            y = df_c[[label]]
        if isinstance(X, np.ndarray):
            dbscan = st.session_state.dbscan
            df_c['anomaly'] = pd.Series(dbscan.predict(X))
            df_c['probability_score'] = pd.Series(norm_score(dbscan.decision_function(X)))
        else:
            dbscan = st.session_state.dbscan
            df_c['anomaly'] = pd.Series(dbscan.predict(X.values))
            df_c['probability_score'] = pd.Series(norm_score(dbscan.decision_function(X.values)))
    else:
        sh = X.shape[0]
        sh = int(sh/20)

        clustering = DBSCAN(eps=1200,min_samples=sh).fit(X)
        st.session_state.dbscan = clustering
        y_pred = clustering.labels_

        y_pred[np.where(y_pred >= 0)] = 0
        y_pred[np.where(y_pred == -1)] = 1
        if use_label:
            y_tr = df_c[label]

            recall = recall_score(y_tr, y_pred, average='macro')
            precision = precision_score(y_tr, y_pred, average='macro')
            f1_sc = f1_scorer(y_tr, y_pred, average='macro')

    return precision, recall, f1_sc, tp, fp, fn, results,eng_fe

def conv_preds(preds):
    '''Convert Predictions to 0 and 1 for easier computation'''
    preds[np.where(preds >= 0)] = 0
    preds[np.where(preds == -1)] = 1
    return preds

def anomaly_detection_pipeline_lof(df, n_comp,window,sensor_cols,inf):
    '''Implementation Local Outlier Factor (LOF) algorithm for anomaly detection'''
    df_c = df.copy()
    X = df_c[list(sensor_cols)]
    st.session_state.df_viz = df_c
    eng_fe = pd.DataFrame()
    use_label, label = check_label()
    precision, recall, f1_score, tp, fp, fn = ['No Labels'] * 6
    p, r, f1, tp, fp, fn, results,f1_sc = [None] * 8

    X = fe_scale_pca(X, window, n_comp, inf, use_label, df_c, label)
    X = X.to_numpy()

    if inf:
        if use_label:
            y = df_c[[label]]
        if isinstance(X, np.ndarray):
            lof = st.session_state.lof
            st.session_state.df_viz_test = df_c
            y_preds = conv_preds(lof.predict(X))
            df_c['anomaly'] = pd.Series(y_preds)
            df_c['probability_score'] = pd.Series(norm_score(lof.decision_function(X)))

        else:
            lof = st.session_state.lof
            st.session_state.df_viz_test = df_c
            y_preds = conv_preds(lof.predict(X.values))
            df_c['anomaly'] = pd.Series(y_preds)
            df_c['probability_score'] = pd.Series(norm_score(lof.decision_function(X.values)))
    else:
        if use_label:
            y_true = df_c[label]
            contamination_rate = np.unique(y_true,return_counts=True)[1][1] / np.unique(y_true,return_counts=True)[1][0]
        else:
            contamination_rate = CONTAMINATION_RATE
        st.session_state.model_type = 'lof'
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_rate,novelty=True).fit(X)
        st.session_state.lof = lof
        y_preds = conv_preds(lof.predict(X))

    if use_label:
        y_true = df_c[label]
        recall = recall_score(y_true, y_preds, average='macro')
        precision = precision_score(y_true, y_preds, average='macro')
        f1_sc = f1_scorer(y_true, y_preds, average='macro')
    if isinstance(X, np.ndarray):
        df_c['anomaly'] = pd.Series(lof.predict(X))
        df_c['probability_score'] = pd.Series(norm_score(lof.decision_function(X)))
    else:
        df_c['anomaly'] = pd.Series(lof.predict(X.values))
        df_c['probability_score'] = pd.Series(norm_score(lof.decision_function(X.values)))

    return precision, recall, f1_sc, tp, fp, fn, df_c,eng_fe


def anomaly_detection_pipeline_sos(df, n_comp,window,sensor_cols,inf):
    '''Implementation Stochastic Outlier Selection (SOS) algorithm for anomaly detection'''
    df_c = df.copy()
    df_c = df_c[10000:15001]

    X = df_c[list(sensor_cols)]
    eng_fe = pd.DataFrame()
    use_label, label = check_label()
    precision, recall, f1_score, tp, fp, fn = ['No Labels'] * 6
    p, r, f1, tp, fp, fn, results,f1_sc = [None] * 8

    X = fe_scale_pca(X, window, n_comp, inf, use_label, df_c, label)
    X.index = df_c.index

    ori_cols = list(df_c.columns)
    X = X.drop_duplicates()
    X_ = X.copy()
    X_['old_ind'] = X_.index

    cdf = X_.join(df_c,on='old_ind')
    df_c = cdf[ori_cols]

    del X_

    if inf:
        st.session_state.pca_test = X
        st.session_state.df_viz_test = df_c
    else:
        st.session_state.pca_train = X
        st.session_state.df_viz = df_c

    if inf:
        if use_label:
            y = df_c[[label]]
        if isinstance(X, np.ndarray):
            sos = st.session_state.sos

            y_preds = conv_preds(sos.predict(X))
            df_c['anomaly'] = pd.Series(y_preds)
            df_c['probability_score'] = pd.Series(sos.decision_function(X))


        else:
            sos = st.session_state.sos
            st.session_state.df_viz_test = df_c
            y_preds = conv_preds(sos.predict(X.values))
            df_c['anomaly'] = pd.Series(y_preds)
            df_c['probability_score'] = pd.Series(sos.decision_function(X.values))
    else:
        st.session_state.model_type = 'sos'

        import pycaret
        if pycaret.__version__ <= '2.2.4':
            setup(data=X, data_split_shuffle=True)
        else:
            setup(data=X)
        sos = create_model('sos',shuffle=True)
        sos_predictions = predict_model(model=sos, data=X)
        y_preds = sos_predictions.Anomaly.to_numpy()
        st.session_state.sos = sos
        print(np.unique(y_preds,return_counts=True))



        if isinstance(X, np.ndarray):
            df_c['anomaly'] = pd.Series(y_preds)
            df_c['probability_score'] = sos_predictions.Anomaly_Score #pd.Series(norm_score(sos.decision_function(X)))
        else:
            df_c['anomaly'] = pd.Series(y_preds)
            df_c['probability_score'] = sos_predictions.Anomaly_Score#pd.Series(norm_score(sos.decision_function(X.values)))


    if use_label:
        y_true = df_c[label]
        recall = recall_score(y_true, y_preds, average='macro')
        precision = precision_score(y_true, y_preds, average='macro')
        f1_sc = f1_scorer(y_true, y_preds, average='macro')




    return precision, recall, f1_sc, tp, fp, fn, df_c,eng_fe



@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def download_csv(df):
    '''Function to download dataframe into CSV'''
    csv = convert_df(df)
    st.download_button(
        "Download file",
        csv,
        "deteced_anomalies.csv",
        "text/csv",
        key='download-csv'
    )


def detect_anomalies():
    st.title('Detect Anomalies')
    if 'df' not in st.session_state or 'sensors' not in st.session_state:
        st.write('Please upload CSV file for anomaly detection and select the sensors you would like to consider for anomaly detection.')

    else:
        df = st.session_state.df
        n_comp = 2 # number of PCA components
        if 'window_size' not in st.session_state:
            window = 5
        else:
            window = st.session_state.window_size

        option = st.selectbox('Anomaly detection Type', (
        'Isolation Forest','Local Outlier Factor (LOF)','Stochastic Outlier Selection (SOS)'))

        sensor_cols = st.session_state.sensors
        if option == 'Isolation Forest':

            if st.button('Start detecting anomalies'):
                with st.spinner('Running anomaly detection. Please hold as this may take some time...'):
                    p, r, f1, tp, fp, fn, results,eng_fe = anomaly_detection_pipeline_iso_fst(df, n_comp=n_comp,window=window,sensor_cols=sensor_cols,inf=False)
                    row_dict = {'precision': [p], 'recall': [r], 'f1_score': [f1]}
                    df_res = pd.DataFrame.from_dict(row_dict)
                    results['anomaly_con'] = results.anomaly.apply(lambda x: 1 if x == -1 else np.nan)

                    for sensor in sensor_cols:
                        sensor_name = sensor + '_a'
                        results[sensor_name] = results[sensor] * results.anomaly_con
                    st.session_state.iso_results = results
                    st.success('Anomalies have been detected!')

                    st.title('Condition Indicators and Detected Anomalies')
                    st.write('Detected anomalies are shown in the last column. True: Values are anomalous, False: Values are normal')
                    eng_fe = eng_fe.reindex(sorted(eng_fe.columns), axis=1)
                    eng_fe.loc[:,'anomaly'] = results.anomaly
                    eng_fe['anomaly'] = eng_fe.anomaly.apply(lambda x: 'True' if x == -1 else 'False')
                    eng_fe['probability_score'] = results.probability_score
                    if 'fe_df' in st.session_state:
                        st.dataframe(st.session_state.fe_df)
                    st.dataframe(eng_fe)

                    download_csv(eng_fe)

                    st.title('Results')
                    if check_label()[0]:
                        st.dataframe(df_res.style.format("{:.2}"))
                    else:
                        st.write('Evaluation is not supported currently as no labels were provided.')
        elif option == 'DBSCAN':
            ## Getting memory error for large datasets. Putting this on hold.
            if st.button('Start detecting anomalies'):
                with st.spinner('Running anomaly detection. Please hold as this may take some time...'):
                    p, r, f1, tp, fp, fn, results,eng_fe = anomaly_detection_pipeline_dbscan(df, n_comp=n_comp,window=window,sensor_cols=sensor_cols,inf=False)
        elif option == 'Local Outlier Factor (LOF)':
            if st.button('Start detecting anomalies'):
                with st.spinner('Running anomaly detection. Please hold as this may take some time...'):
                    p, r, f1, tp, fp, fn, results,eng_fe = anomaly_detection_pipeline_lof(df, n_comp=n_comp,window=window,sensor_cols=sensor_cols,inf=False)
                    row_dict = {'precision': [p], 'recall': [r], 'f1_score': [f1]}
                    df_res = pd.DataFrame.from_dict(row_dict)
                    print(df_res)
                    st.write(df_res)
                    results['anomaly_con'] = results.anomaly.apply(lambda x: 1 if x == -1 else np.nan)
                    for sensor in sensor_cols:
                        sensor_name = sensor + '_a'
                        results[sensor_name] = results[sensor] * results.anomaly_con
                    st.session_state.lof_results = results
                    st.success('Anomalies have been detected!')

                    st.title('Condition Indicators and Detected Anomalies')
                    st.write('Detected anomalies are shown in the last column. True: Values are anomalous, False: Values are normal')
                    eng_fe = eng_fe.reindex(sorted(eng_fe.columns), axis=1)
                    eng_fe.loc[:,'anomaly'] = results.anomaly
                    eng_fe['anomaly'] = eng_fe.anomaly.apply(lambda x: 'True' if x == -1 else 'False')
                    eng_fe['probability_score'] = results.probability_score
                    if 'fe_df' in st.session_state:
                        st.dataframe(st.session_state.fe_df)

                    st.dataframe(eng_fe)

                    download_csv(eng_fe)

                    st.title('Results')
                    if check_label()[0]:
                        st.dataframe(df_res.style.format("{:.2}"))
                    else:
                        st.write('Evaluation is not supported currently as no labels were provided.')
        elif option == 'Stochastic Outlier Selection (SOS)':
            if st.button('Start detecting anomalies'):
                with st.spinner('Running anomaly detection. Please hold as this may take some time...'):
                    p, r, f1, tp, fp, fn, results,eng_fe = anomaly_detection_pipeline_sos(df, n_comp=n_comp,window=window,sensor_cols=sensor_cols,inf=False)

                    row_dict = {'precision': [p], 'recall': [r], 'f1_score': [f1]}
                    df_res = pd.DataFrame.from_dict(row_dict)
                    st.write(df_res)
                    results['anomaly_con'] = results.anomaly.apply(lambda x: 1 if x == -1 else np.nan)
                    for sensor in sensor_cols:
                        sensor_name = sensor + '_a'
                        results[sensor_name] = results[sensor] * results.anomaly_con
                    st.session_state.sos_results = results
                    st.success('Anomalies have been detected!')

                    st.title('Condition Indicators and Detected Anomalies')
                    st.write('Detected anomalies are shown in the last column. True: Values are anomalous, False: Values are normal')
                    eng_fe = eng_fe.reindex(sorted(eng_fe.columns), axis=1)
                    eng_fe.loc[:,'anomaly'] = results.anomaly
                    eng_fe['anomaly'] = eng_fe.anomaly.apply(lambda x: 'True' if x == -1 else 'False')
                    eng_fe['probability_score'] = results.probability_score
                    if 'fe_df' in st.session_state:
                        st.dataframe(st.session_state.fe_df)

                    st.dataframe(eng_fe)

                    download_csv(eng_fe)

                    st.title('Results')
                    if check_label()[0]:
                        st.dataframe(df_res.style.format("{:.2}"))
                    else:
                        st.write('Evaluation is not supported currently as no labels were provided.')







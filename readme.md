This repository is used for multivariate anomaly detection in time series data. It is a Proof-of-Concept on anomaly detection. 

This anomaly detector derives condition indicators such as Standard Deviation, Mean, Kurtosis, Variance and Max using a rolling window of size 5. The derived features are then condensed into 5 components using Principal Component Analysis. These features are the fed into Isolation forest for anomalous points to be predicted. To begin detecting anomalous points in your multivariate time series, proceed to upload.


**How to use:**

Upload multivariate time series.
Once CSV is uploaded, click on Start Anomaly Detection
This will take ~1-2 minutes. Post completion, click on See Anomalous Points for the Red points in the time series chart. These represent the outliers.
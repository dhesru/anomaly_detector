This repository is used for multivariate anomaly detection in time series data. It is a Proof-of-Concept on anomaly detection. 

This anomaly detector derives condition indicators such as Standard Deviation, Mean, Kurtosis, Variance and Max using a rolling window of size between 5 and 15 (customizable). The derived features are then condensed into 5 components using Principal Component Analysis. These features are the fed into model of choice (Isolation Forest, Local Outlier Factor (LOF), Stochastic Outlier Selection (SOS)) for anomalous points to be predicted. To begin detecting anomalous points in your multivariate time series, proceed to upload.


**Some tips on usage**

**Upload Data:**

This tab is used for uploading of Sensor readings for anomaly detection. Once you have uploaded the CSV, select the sensor readings that needs to be utilized.

**Train & Detect Anomalies**

This tab is used to detect anomalies. You may chose your choice of algorithm to be used for anomaly detection. Once the anomalies are detected, the condition indicators and evaluation metrics will be displayed. 

**Visualization of Results**

This tab is used for visualization of the results. It displays the Principal Compoenents on the data and anomalous points on time series data for both training and inferenced data.

**Model Inference**

This tab is used for detecting anomalies on new dataset. The model which was trained last, shall be utilized in the inferencing.
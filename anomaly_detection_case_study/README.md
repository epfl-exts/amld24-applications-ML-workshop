<img src="../static/logo_red.png" width="125px" align="right">


## Anomaly Detection

This repository provides the resources for the session on **Anomaly Detection** at the **EPFL Extension School Workshop - Applications in Machine Learning**.

**Dataset**

The data is based on the [KDD-CUP 1999 challenge](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) on network intrusion detection. A description of the original task can be found [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html). The data provided for this workshop has been adapted from the [NSL-KDD version](https://www.kaggle.com/hassan06/nslkdd).

**Anomaly detection**

Anomaly detection can be treated as a supervised classification task. However this approach struggles when the portion of anomalies (here network attacks) is small. Instead we showcase an approach using [Isolation Forests](https://www.youtube.com/watch?v=RyFQXQf4w4w). 

The user can select the size of training dataset and vary its contamination rate, including a dataset without any anomalies. The model is then trained on this dataset, used to predict anomalies on a separate test set and evaluate the performance.


## Hands-On Session

To get started with the hands-on session you have the following options. You can choose the first three by clicking on the badges below:

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epfl-exts/amld24-applications-ML-workshop/blob/main/anomaly_detection_case_study/anomalies_detection_interactive.ipynb) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epfl-exts/amld24-applications-ML-workshop/c03e8f694cde6e00615d8c340f2ee93fa512f816?urlpath=lab%2Ftree%2Fanomaly_detection_case_study%2Fanomalies_detection_interactive.ipynb)
[![Offline](https://img.shields.io/badge/Offline_View-Open-Blue.svg)](https://github.com/epfl-exts/amld24-applications-ML-workshop/blob/main/static/anomalies_detection_completed.ipynb)

- Open the jupyter notebook in **Google Colab** to run the codes interactively on the cloud (recommended for this workshop). Note that you need to have a Google account to run the code in Google Colab.

- You can also interactively run the codes on a server using **Binder**. If you don't have a Google account, you can use this option. 

- You can choose to take a look at the already executed notebook in the **Offline View**. Note that with this option you cannot run the codes interactively.

- Lastly, if you want to run the codes locally on your machine you can follow the steps outlined [here](https://github.com/epfl-exts/amld24-applications-ML-workshop/blob/main/README.md). 

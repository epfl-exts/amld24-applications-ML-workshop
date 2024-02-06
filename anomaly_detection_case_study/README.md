<img src="static/EXTS_Logo.png" width="125px" align="right">

# Application in Machine Learning: Anomaly Detection

This repository provides the resources for the session and accompanying hands-on exercises on **Anomaly Detection** at the **EPFL Extension School Workshop - Applications in Machine Learning**.

**Dataset**

The data is based on the [KDD-CUP 1999 challenge](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) on network intrusion detection. A description of the original task can be found [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html). The data provided for this workshop has been adapted from the [NSL-KDD version](https://www.kaggle.com/hassan06/nslkdd).

**Anomaly detection**

Anomaly detection can be treated as a supervised classification task. However this approach struggles when the portion of anomalies (here network attacks) is small. Instead we showcase an approach using [Isolation Forests](https://www.youtube.com/watch?v=RyFQXQf4w4w). 

The user can select the size of training dataset and vary its contamination rate, including a dataset without any anomalies. The model is then trained on this dataset, used to predict anomalies on a separate test set and evaluate the performance.

**Getting started:** 

Start with loading settings and functions. If you want to execute a cell, make sure it is selected and then press `SHIFT`+`ENTER` or the `'Play'` button.

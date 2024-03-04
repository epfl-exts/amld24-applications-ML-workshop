
<img src="static/logo_red.png" width="150px" align="right">

# Applications of Machine Learning workshop


Welcome to the [Applications of Machine Learning workshop]()! This repository contains all resources for the workshop presented at the [Applied Machine Learning Days 2024](https://2024.appliedmldays.org/). Whether you're a student, professional, or simply curious about the field, this workshop will provide you with a solid foundation in key machine learning concepts and practical applications.


## Overview


The workshop is divided into four parts:

1. **Introduction**: This section provides a comprehensive introduction to machine learning, explaining fundamental concepts and laying the groundwork for the hands-on sessions. You can find presentation slides [here]().

2. **Classification**: In this part you will learn about text classification as a supervised learning problem. You will dive into the realm of natural language processing and learn how machines can be trained to analyze text data. Here is the [text classification repository](https://github.com/epfl-exts/amld24-applications-ML-workshop/tree/main/text_classification_case_study)! Enjoy learning.

3. **Regression**: In this third part you will learn about timeseries regression as an another supervised learning problem. You will understand the significance of time dimension in data and how machine learning can be applied to timeseries forecasting. Jump into the [timeseries regression repository](https://github.com/epfl-exts/amld24-applications-ML-workshop/tree/main/timeseries_regression_case_study) and start learning.

4. **Anomaly Detection**: Here you will explore the intriguing field of anomaly detection, where machine learning is used to identify unusual patterns or outliers in data. Explore the [anomaly detection repository](https://github.com/epfl-exts/amld24-applications-ML-workshop/tree/main/anomaly_detection_case_study) and enjoy learning about anomaly detection as a supervised classification problem.



## Getting started

To get started there are two options:

### Run Hands-On in the Cloud (recommended)

The most straightforward way to run hands-on exercises is to execute them in the cloud. For each of the three parts, open the jupyter notebook in Google Colab to run the code interactively:

<a href="https://colab.research.google.com/github/epfl-exts/amld24-applications-ML-workshop/blob/main/timeseries_regression_case_study/timeseries_prediction_workbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Text Classification: Open in Colab"/></a>

[![Text Classification](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epfl-exts/amld24-applications-ML-workshop/blob/main/timeseries_regression_case_study/timeseries_prediction_workbook.ipynb)

[![Timeseries Regression](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epfl-exts/amld24-applications-ML-workshop/blob/main/timeseries_regression_case_study/timeseries_prediction_workbook.ipynb)

[![Anomaly Detection](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epfl-exts/amld24-applications-ML-workshop/blob/main/timeseries_regression_case_study/timeseries_prediction_workbook.ipynb)

### Run Hands-On locally on your machine

Should you prefer to run the hands-on locally on your machine, there are three steps to follow:

1. **Clone the content**: Clone this repository from Github to your local machine using the following command in your terminal:

```
git clone https://github.com/epfl-exts/amld24-applications-ML-workshop.git
```

Or if you prefer to download the content manually, you can use the the green `Code` button on the top right of this page.

2. **Install Miniconda**: Once the content of the repository is on your machine, you can install the relevant Python dependencies with `conda`. But before that you need to install `Miniconda` on your system, if you don't have `conda` installed already. Install Miniconda on your system using this [link](https://docs.conda.io/en/latest/miniconda.html)


and use the provided [environment.yml](https://github.com/epfl-exts/amld24-applications-ML-workshop/blob/main/environment.yml) file.

3. **Installation with conda**: To install the relevant Python dependencies with conda, use the following code in your terminal:

```
conda env create -f ~/amld24-applications-ML-workshop/environment.yml
```

***Note***: This assumes that the downloaded github repository was stored in your home folder.
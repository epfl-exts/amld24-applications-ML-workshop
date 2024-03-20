<img src="../static/logo_red.png" width="125px" align="right">


## Time Series Forecasting

This repository provides the resources for the session on **Time Series Forecasting** at the **EPFL Extension School Workshop - Applications in Machine Learning**.

**Dataset**

We use the data from the [Global Energy Forecasting Competition](https://en.wikipedia.org/wiki/Global_Energy_Forecasting_Competition). We focus on the data for the year 2014 which contains 8'760 observations.

**Modeling**

In general, we can define a forecasting problem as a supervised regression or classification task. In this repository we define this problem as a regression task and solve it using two simple and popular models: Linear Regression and Random Forest. There are a few parameters that the user can vary during the data exploration and modeling. After training the model, the user can evaluate the performance on a separate test set and compare it with a simple baseline.

## Hands-On Session

To get started with the hands-on session you have the following options. You can choose the first three by clicking on the badges below:

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmirKhalilzadeh/tmp/blob/main/timeseries_regression_case_study/timeseries_regression_interactive.ipynb) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AmirKhalilzadeh/tmp/5f07b71e2dd2b5ad3055a5d8187f291878adfe5c?urlpath=lab%2Ftree%2Ftimeseries_regression_case_study%2Ftimeseries_regression_interactive.ipynb)
[![Offline](https://img.shields.io/badge/Offline_View-Open-Blue.svg)](https://github.com/AmirKhalilzadeh/tmp/blob/main/static/timeseries_regression_completed.ipynb)

- Open the jupyter notebook in **Google Colab** to run the codes interactively on the cloud (recommended for this workshop). Note that you need to have a Google account to run the code in Google Colab.

- You can also interactively run the codes on a server using **Binder**. If you don't have a Google account, you can use this option. 

- You can choose to take a look at the already executed notebook in the **Offline View**. Note that with this option you cannot run the codes interactively.

- Lastly, if you want to run the codes locally on your machine you can follow the steps outlined [here](https://github.com/epfl-exts/amld24-applications-ML-workshop/blob/main/README.md). 
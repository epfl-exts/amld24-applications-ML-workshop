<img src="../static/logo_red.png" width="125px" align="right">


## Text Classification

This repository provides the resources for the session on **Text Classification** at the **EPFL Extension School Workshop - Applications in Machine Learning**.

**Dataset**

We will use the [SpamAssassin](https://spamassassin.apache.org/) public email corpus. This dataset contains ~6'000 labeled emails with a ~30% spam ratio. The dataset has been downloaded for you and is available in the data folder. 

**Modeling**

We want to build a spam detector which, given examples of spam emails and examples of regular emails, learns how to flag new emails as spam or non-spam. In this repository we define this problem as a classification task and solve it using Logistic Regression classifier. After training the model, the user can evaluate the model performance on a separate test set and compare it with a simple baseline.

## Hands-On Session

To get started with the hands-on session you have the following options. You can choose the first three by clicking on the badges below:


[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmirKhalilzadeh/tmp/blob/main/text_classification_case_study/Text_classification_interactive.ipynb) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AmirKhalilzadeh/tmp/5f07b71e2dd2b5ad3055a5d8187f291878adfe5c?urlpath=lab%2Ftree%2Ftext_classification_case_study%2FText_classification_interactive.ipynb)
[![Offline](https://img.shields.io/badge/Offline_View-Open-Blue.svg)](https://github.com/AmirKhalilzadeh/tmp/blob/main/static/Text_classification_completed.ipynb)


- Open the jupyter notebook in **Google Colab** to run the codes interactively on the cloud (recommended for this workshop). Note that you need to have a Google account to run the code in Google Colab.

- You can also interactively run the codes on a server using **Binder**. If you don't have a Google account, you can use this option. 

- You can choose to take a look at the already executed notebook in the **Offline View**. Note that with this option you cannot run the codes interactively.

- Lastly, if you want to run the codes locally on your machine you can follow the steps outlined [here](https://github.com/epfl-exts/amld24-applications-ML-workshop/blob/main/README.md). 
# 💻 Laptop Price Prediction (ML)

Predict laptop prices (€) from specifications using machine learning.  
Trains baseline and ensemble models, evaluates with MAE/MSE/R², and includes a Streamlit app for interactive inference.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)]()
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red)]()

---

## 📓 Notebooks

**EDA Notebook**  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NoellaButi/ai-ml-laptop-price/blob/main/notebooks/01_eda.ipynb)

**Modeling Notebook**  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NoellaButi/ai-ml-laptop-price/blob/main/notebooks/02_modeling.ipynb)

---

## 🚀 Live Demo

<a href="https://laptop-prediction-prices.streamlit.app/" target="_blank">Explore the deployed app on Streamlit</a>

---

## 🔹 Features

- 📂 **Dataset**: raw laptop specs stored in `data/raw/laptop_prices.csv`
- 📊 **Exploratory Data Analysis (EDA)**: performed in `notebooks/01_eda.ipynb`
- 🤖 **Model Training & Evaluation**: Linear Regression, Random Forest, and Gradient Boosting (`notebooks/02_modeling.ipynb`)
- 📝 **Reports & Metrics**: results saved in `reports/train_report.json` and `reports/assets/metrics.json`
- 💾 **Artifacts**: trained models and best estimator stored in `artifacts/`
- 🌐 **Interactive App**: Streamlit app (`src/app.py`) for single or batch price prediction
- 🔄 **Reproducibility**: environment captured in `requirements.txt` and open-source licensed

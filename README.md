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
- 💾 **Artifacts**: trained models and best estimator stored in `artifacts/best_gb_model.pkl` and `artifacts/model_gbr.pkl`
- 🌐 **Interactive App**: Streamlit app (`src/app.py`) for single or batch price prediction
- 🔄 **Reproducibility**: environment captured in `requirements.txt` and open-source licensed

---

## 🔹 Results

Evaluated on a held-out test split (**test_size = 0.2**, **random_state = 42**).

| Model                     | MSE       | RMSE   | R²    |
|----------------------------|----------:|-------:|------:|
| Linear Regression          | 172,467   | 415.29 | 0.653 |
| Random Forest              | 87,042    | 295.03 | 0.825 |
| Gradient Boosting (numeric)| 81,011    | 284.63 | 0.837 |
| **Gradient Boosting (full)** | **58,510** | **241.89** | **0.882** |
| **Final GB (best CV)**     | —         | **237.80** | **0.886** |

📊 Detailed artifacts are saved in:
- `reports/assets/metrics.json`
- `reports/train_report.json`

### 🔹 Best Model (Cross-Validation)
- **Gradient Boosting (full feature set)**
- Best hyperparameters:
  - `learning_rate = 0.1`
  - `max_depth = 3`
  - `n_estimators = 300`

### 🔹 Features Used
- **Numeric:** `Ram`, `Weight`, `CPU_freq`, `PrimaryStorage`
- **Categorical:** `Company`, `CPU_company`, `CPU_model`, `GPU_company`, `GPU_model`, `OS`, `TypeName`, `Screen`, `PrimaryStorageType`, `SecondaryStorageType`, `Touchscreen`, `IPSpanel`, `RetinaDisplay`

---

## 🔹 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/NoellaButi/ai-ml-laptop-price.git
   cd ai-ml-laptop-price

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python src/train.py
   ```

5. **Launch the app**
   ```bash
   streamlit run src/app.py
   ```

---

## 🔹 Project Structure

ai-ml-laptop-price/
├─ artifacts/ # trained models & saved artifacts
│ ├─ .gitkeep
│ ├─ best_gb_model
│ └─ model_gbr
│
├─ data/
│ ├─ raw/
│ │ ├─ .gitkeep
│ │ └─ laptop_prices.csv
│ └─ preprocessed/ # (reserved; no preprocessing used yet)
│
├─ notebooks/
│ ├─ .gitkeep
│ ├─ 01_eda.ipynb
│ └─ 02_modeling.ipynb
│
├─ reports/
│ ├─ train_report.json
│ └─ assets/
│ └─ metrics.json
│
├─ src/
│ ├─ app.py # Streamlit app (single/batch prediction)
│ ├─ train.py # training script
│ └─ test.py # tests or quick experiments
│
├─ requirements.txt
├─ LICENSE
└─ README.md



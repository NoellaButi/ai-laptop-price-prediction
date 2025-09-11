# Laptop Price Prediction 💻💶  
Predict Laptop Prices (€) from Specifications using Machine Learning  

![Language](https://img.shields.io/badge/language-Python-blue.svg) 
![Notebook](https://img.shields.io/badge/tool-Jupyter-orange.svg) 
![App](https://img.shields.io/badge/app-Streamlit-red.svg) 
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  

---

✨ **Overview**  
This project predicts laptop prices in Euros (€) from their specifications using regression and ensemble methods.  
It includes exploratory data analysis, model training, and an interactive **Streamlit app** for single or batch predictions.  

🛠️ **Workflow**  
- Load raw dataset of laptop specifications  
- Perform EDA (distributions, correlations, outliers)  
- Train baseline (Linear Regression) and ensemble models (Random Forest, Gradient Boosting)  
- Save trained models & metrics as artifacts  
- Deploy interactive app with Streamlit  

📁 **Repository Layout**  
```bash
data/           # raw & preprocessed datasets
notebooks/      # EDA and modeling notebooks
src/            # training script & Streamlit app
reports/        # metrics, plots, training results
artifacts/      # saved models
requirements.txt
README.md
```

🚦 **Demo**

Train model:
```bash
python src/train.py
```

Run Streamlit app:
```bash
streamlit run src/app.py
```

🔍 **Features**
- Exploratory Data Analysis with visualizations
- Regression models: Linear, Random Forest, Gradient Boosting
- Evaluation: MAE, MSE, R²
- Saves trained models & reports as artifacts
- Interactive Streamlit app for inference

🚦 **Results (Held-Out Test Set)**
```bash
Model                MSE      RMSE     R²
--------------------------------------------
Linear Regression   172,467   415.29   0.653
Random Forest        87,042   295.03   0.825
Gradient Boosting    58,510   241.89   0.882
Final GB (CV best)     —      237.80   0.886
```

📜 **License**

MIT (see [LICENSE](LICENSE))

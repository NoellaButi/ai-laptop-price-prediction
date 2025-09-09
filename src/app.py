# src/app.py
"""
Streamlit UI for Laptop Price Prediction

Run from project root:
    streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json
import glob

import joblib
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

# -----------------------------
# Project paths (robust)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR   = PROJECT_ROOT / "reports"
DATA_CSV      = PROJECT_ROOT / "data" / "raw" / "laptop_prices.csv"
ASSETS_DIR    = PROJECT_ROOT / "assets"
LOGO_PATH     = ASSETS_DIR / "logo.png"
PAGE_ICON     = str(LOGO_PATH) if LOGO_PATH.exists() else "💻"

# Keep in sync with train/test scripts
FEATURE_NUMERIC = ["Ram", "Weight", "CPU_freq", "PrimaryStorage"]
FEATURE_CATS = [
    "Company","CPU_company","CPU_model","GPU_company","GPU_model","OS",
    "TypeName","Screen","PrimaryStorageType","SecondaryStorageType",
    "Touchscreen","IPSpanel","RetinaDisplay",
]
ALL_FEATURES = FEATURE_NUMERIC + FEATURE_CATS

# -----------------------------
# Presets to speed up demos
# -----------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    "Balanced Ultrabook": {
        "Ram": 16, "Weight": 1.3, "CPU_freq": 2.6, "PrimaryStorage": 512,
        "Company": "Dell", "CPU_company": "Intel", "CPU_model": "Core i7-1165G7",
        "GPU_company": "Intel", "GPU_model": "Iris Xe", "OS": "Windows 11",
        "TypeName": "Ultrabook", "Screen": "Full HD",
        "PrimaryStorageType": "SSD", "SecondaryStorageType": "None",
        "Touchscreen": "No", "IPSpanel": "Yes", "RetinaDisplay": "No",
    },
    "Budget Notebook": {
        "Ram": 8, "Weight": 1.6, "CPU_freq": 2.0, "PrimaryStorage": 256,
        "Company": "Acer", "CPU_company": "Intel", "CPU_model": "Core i3",
        "GPU_company": "Intel", "GPU_model": "UHD Graphics", "OS": "Windows 10",
        "TypeName": "Notebook", "Screen": "Standard",
        "PrimaryStorageType": "SSD", "SecondaryStorageType": "None",
        "Touchscreen": "No", "IPSpanel": "No", "RetinaDisplay": "No",
    },
    "Creator Laptop": {
        "Ram": 32, "Weight": 2.0, "CPU_freq": 3.2, "PrimaryStorage": 1024,
        "Company": "ASUS", "CPU_company": "AMD", "CPU_model": "Ryzen 9",
        "GPU_company": "NVIDIA", "GPU_model": "RTX 4070", "OS": "Windows 11",
        "TypeName": "Gaming", "Screen": "4K Ultra HD",
        "PrimaryStorageType": "SSD", "SecondaryStorageType": "SSD",
        "Touchscreen": "No", "IPSpanel": "Yes", "RetinaDisplay": "No",
    },
}

# -----------------------------
# Utils
# -----------------------------
def eur_to_usd(amount_eur: float, rate: float = 1.08) -> float:
    """Tiny FX helper; rate is user-adjustable."""
    return float(amount_eur) * float(rate)

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    return joblib.load(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def list_models(artifacts_dir: Path) -> List[Path]:
    """Find available .pkl models in artifacts/."""
    paths = [Path(p) for p in glob.glob(str(artifacts_dir / "*.pkl"))]
    return sorted(paths)

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns exist and numeric cols are coerced."""
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = None
    df = df[ALL_FEATURES]
    for c in FEATURE_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def predict(model, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(X), index=X.index)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon=PAGE_ICON,
    layout="wide"
)

# Small CSS polish
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.dataframe thead th { font-weight: 600; }
.small-caption { color: #9ca3af; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("💻 Laptop Price Predictor")
st.markdown('<p class="small-caption">Enter specs or upload a CSV to estimate price. Model: Gradient Boosting Regressor (GBR).</p>', unsafe_allow_html=True)

# -----------------------------
# Sidebar — config & model info
# -----------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_column_width=True)
    else:
        st.write("**Laptop Price Predictor**")
    st.caption("Gradient Boosting • EUR/USD toggle • CSV batch")
    st.divider()

    st.header("Configuration")
    use_usd = st.toggle("Show prices in USD", value=False)
    usd_rate = st.number_input("EUR → USD rate", min_value=0.5, max_value=2.0, value=1.08, step=0.01)

    st.subheader("Model")
    models = list_models(ARTIFACTS_DIR)
    if not models:
        st.error(f"No model found in {ARTIFACTS_DIR}. Run `python src/train.py` first.")
        st.stop()

    default_idx = max(0, next((i for i, p in enumerate(models) if p.name == "model_gbr.pkl"), 0))
    selected_model_path = st.selectbox(
        "Choose a model file",
        options=models,
        index=default_idx,
        format_func=lambda p: p.name
    )

    model = load_model(selected_model_path)
    if model is None:
        st.error(f"Could not load model at {selected_model_path}")
        st.stop()

    st.success(f"Loaded: {selected_model_path.name}")

    st.subheader("Training Report")
    report = load_report(REPORTS_DIR / "train_report.json")
    if report:
        st.json(report)
    else:
        st.info("Report not found (will be created by `train.py`).")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔹 Single Prediction", "📁 Batch (CSV)", "📈 Feature Importance"])

# -----------------------------
# Tab 1 — Single Prediction
# -----------------------------
with tab1:
    st.subheader("Enter laptop specs")
    preset = st.selectbox("Quick preset", ["(None)"] + list(PRESETS.keys()))
    prefill = PRESETS.get(preset, {})

    with st.form("single_predict"):
        col1, col2 = st.columns(2)
        with col1:
            ram = st.number_input("RAM (GB)", 2, 256, int(prefill.get("Ram", 16)), step=2)
            weight = st.number_input("Weight (kg)", 0.5, 6.0, float(prefill.get("Weight", 1.35)), step=0.05, format="%.2f")
            cpu_freq = st.number_input("CPU Frequency (GHz)", 0.8, 6.0, float(prefill.get("CPU_freq", 2.8)), step=0.1, format="%.2f")
            primary_storage = st.number_input("Primary Storage (GB)", 64, 8192, int(prefill.get("PrimaryStorage", 512)), step=64)

        with col2:
            company = st.text_input("Company", prefill.get("Company", "Dell"))
            cpu_company = st.selectbox("CPU Company", ["Intel","AMD","Apple","Other"],
                                       index=["Intel","AMD","Apple","Other"].index(prefill.get("CPU_company","Intel")))
            cpu_model = st.text_input("CPU Model", prefill.get("CPU_model", "Core i7-1165G7"))
            gpu_company = st.selectbox("GPU Company", ["NVIDIA","AMD","Intel","Apple","Other"],
                                       index=["NVIDIA","AMD","Intel","Apple","Other"].index(prefill.get("GPU_company","Intel")))
            gpu_model = st.text_input("GPU Model", prefill.get("GPU_model", "Iris Xe"))

        col3, col4, col5 = st.columns(3)
        with col3:
            os_name = st.selectbox("Operating System", ["Windows 11","Windows 10","Linux","macOS","Other"],
                                   index=["Windows 11","Windows 10","Linux","macOS","Other"].index(prefill.get("OS","Windows 11")))
            typename = st.selectbox("Type", ["Ultrabook","Notebook","Gaming","2 in 1 Convertible","Workstation","Other"],
                                    index=["Ultrabook","Notebook","Gaming","2 in 1 Convertible","Workstation","Other"].index(prefill.get("TypeName","Ultrabook")))
        with col4:
            screen = st.selectbox("Screen", ["Standard","Full HD","Quad HD+","4K Ultra HD","Other"],
                                  index=["Standard","Full HD","Quad HD+","4K Ultra HD","Other"].index(prefill.get("Screen","Full HD")))
            primary_storage_type = st.selectbox("Primary Storage Type", ["SSD","HDD","Flash Storage","Hybrid","Other"],
                                                index=["SSD","HDD","Flash Storage","Hybrid","Other"].index(prefill.get("PrimaryStorageType","SSD")))
        with col5:
            secondary_storage_type = st.selectbox("Secondary Storage Type", ["None","SSD","HDD","Hybrid","Other"],
                                                  index=["None","SSD","HDD","Hybrid","Other"].index(prefill.get("SecondaryStorageType","None")))
            touchscreen = st.selectbox("Touchscreen", ["No","Yes"],
                                       index=["No","Yes"].index(prefill.get("Touchscreen","No")))
            ips = st.selectbox("IPS Panel", ["Yes","No"],
                               index=["Yes","No"].index(prefill.get("IPSpanel","Yes")))
            retina = st.selectbox("Retina Display", ["No","Yes"],
                                  index=["No","Yes"].index(prefill.get("RetinaDisplay","No")))

        # Friendly validations
        if ram > 32 and weight < 1.1:
            st.warning("Ultra-light laptops rarely pair with >32 GB RAM.")
        if primary_storage > 2048:
            st.info("Storage > 2 TB is uncommon for ultrabooks; predictions may extrapolate.")

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        row = {
            "Ram": ram, "Weight": weight, "CPU_freq": cpu_freq, "PrimaryStorage": primary_storage,
            "Company": company, "CPU_company": cpu_company, "CPU_model": cpu_model,
            "GPU_company": gpu_company, "GPU_model": gpu_model, "OS": os_name,
            "TypeName": typename, "Screen": screen, "PrimaryStorageType": primary_storage_type,
            "SecondaryStorageType": secondary_storage_type, "Touchscreen": touchscreen,
            "IPSpanel": ips, "RetinaDisplay": retina,
        }
        df_one = ensure_schema(pd.DataFrame([row]))
        price_eur = predict(model, df_one).iloc[0]
        if use_usd:
            st.metric("Predicted Price (USD)", f"{eur_to_usd(price_eur, usd_rate):,.2f}")
        else:
            st.metric("Predicted Price (EUR)", f"{price_eur:,.2f}")
        with st.expander("See the exact row we predicted on"):
            st.dataframe(df_one, use_container_width=True)

# -----------------------------
# Tab 2 — Batch CSV
# -----------------------------
with tab2:
    st.subheader("Upload a CSV for batch predictions")
    st.caption("Your CSV may include any subset of the following columns; missing columns are auto-filled:\n"
               f"`{', '.join(ALL_FEATURES)}`")
    uploaded = st.file_uploader("Choose a CSV", type=["csv"])

    if uploaded:
        try:
            df_in = ensure_schema(pd.read_csv(uploaded))
            preds = predict(model, df_in)
            out = df_in.copy()
            out["Predicted_Price_EUR"] = preds
            if use_usd:
                out["Predicted_Price_USD"] = out["Predicted_Price_EUR"].apply(lambda x: eur_to_usd(x, usd_rate))
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out, use_container_width=True)

            st.download_button(
                "Download predictions as CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="laptop_price_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# -----------------------------
# Tab 3 — Feature Importance
# -----------------------------
with tab3:
    st.subheader("Permutation Importance (approximate)")
    st.caption("Runs on a small sample; shows drop in R² when a feature is shuffled.")
    if not DATA_CSV.exists():
        st.info("`data/raw/laptop_prices.csv` not found — run training or place the CSV to enable this.")
    else:
        n_sample = st.slider("Sample size", 100, 1000, 400, step=50)
        n_repeats = st.slider("Repeats", 3, 15, 5, step=1)
        go = st.button("Compute Importance")
        if go:
            try:
                df_imp = pd.read_csv(DATA_CSV).head(n_sample)
                for c in ALL_FEATURES:
                    if c not in df_imp.columns:
                        df_imp[c] = None
                for c in FEATURE_NUMERIC:
                    df_imp[c] = pd.to_numeric(df_imp[c], errors="coerce")
                mask = df_imp["Price_euros"].notna()
                X_imp = df_imp.loc[mask, ALL_FEATURES]
                y_imp = pd.to_numeric(df_imp.loc[mask, "Price_euros"], errors="coerce")

                result = permutation_importance(
                    model, X_imp, y_imp,
                    n_repeats=n_repeats, random_state=42, n_jobs=-1, scoring="r2"
                )
                importances = pd.DataFrame({
                    "feature": ALL_FEATURES,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }).sort_values("importance_mean", ascending=False).head(15)

                st.dataframe(importances, use_container_width=True)
            except Exception as e:
                st.error(f"Could not compute permutation importance: {e}")
"""
train.py — Train a laptop price prediction model end-to-end.

What it does
------------
1) Loads the dataset from data/raw/laptop_prices.csv
2) Builds a preprocessing + model pipeline (numeric imputation + one-hot + GBR)
3) Tunes a few hyperparameters with GridSearchCV
4) Evaluates on a hold-out test split (RMSE, R^2)
5) Saves:
   - Best model → artifacts/model_gbr.pkl
   - Training report (JSON) → reports/train_report.json

Why it’s robust
---------------
- Paths resolved from this file’s location (works from IDE/terminal/CI)
- Defensive checks for required columns and file existence
- Comments explain each step for a broad audience
"""

from __future__ import annotations

from pathlib import Path
from math import sqrt
import sys
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Project paths (robust)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH     = PROJECT_ROOT / "data" / "raw" / "laptop_prices.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR   = PROJECT_ROOT / "reports"

# Create output directories if they don't exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Configuration
# -----------------------------
# Feature names used during training (keep these in sync with your data)
FEATURE_NUMERIC = ["Ram", "Weight", "CPU_freq", "PrimaryStorage"]
REQUIRED_NUMERIC = FEATURE_NUMERIC + ["Price_euros"]  # includes target for validation

FEATURE_CATS = [
    "Company", "CPU_company", "CPU_model", "GPU_company", "GPU_model", "OS",
    "TypeName", "Screen", "PrimaryStorageType", "SecondaryStorageType",
    "Touchscreen", "IPSpanel", "RetinaDisplay",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Small, sensible grid (expand later if you need more performance)
PARAM_GRID = {
    "model__n_estimators": [150, 300],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3],
}


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and do basic sanity checks/coercions."""
    print(f"Loading dataset from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"Couldn't find CSV at: {path}")

    df = pd.read_csv(path)

    # Ensure required numeric columns exist and are numeric where possible
    missing_required = [c for c in REQUIRED_NUMERIC if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    for c in REQUIRED_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Limit to known categorical columns that are present in the dataset
    present_cats = [c for c in FEATURE_CATS if c in df.columns]

    return df, present_cats


def build_pipeline(feature_numeric: list[str], feature_cats: list[str]) -> Pipeline:
    """Create a preprocessing + model pipeline.

    - Numeric: median imputation
    - Categorical: most frequent imputation + one-hot encode
    - Model: GradientBoostingRegressor
    """
    numeric_tf = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, feature_numeric),
            ("cat", categorical_tf, feature_cats),
        ],
        remainder="drop",
    )

    gbr = GradientBoostingRegressor(random_state=RANDOM_STATE)

    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("model", gbr),
        ]
    )
    return pipe


def train_and_evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[GridSearchCV, float, float]:
    """Split data, run GridSearchCV, compute metrics, and return (grid, rmse, r2)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Use R^2 for model selection; we'll still report RMSE for interpretability
    grid = GridSearchCV(
        pipe,
        PARAM_GRID,
        cv=3,
        n_jobs=-1,
        scoring="r2",
        verbose=1,
    )
    grid.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = grid.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)  # safer than using squared=False for broad sklearn versions
    r2 = r2_score(y_test, y_pred)

    return grid, rmse, r2


def save_artifacts(grid: GridSearchCV, rmse: float, r2: float, feature_cats: list[str]) -> None:
    """Persist best model and a training report to disk."""
    model_path = ARTIFACTS_DIR / "model_gbr.pkl"
    joblib.dump(grid.best_estimator_, model_path)

    report = {
        "best_params": grid.best_params_,
        "rmse": rmse,
        "r2": r2,
        "features": {
            "numeric": FEATURE_NUMERIC,
            "categorical": feature_cats,
        },
        "data_path": str(DATA_PATH),
    }
    (REPORTS_DIR / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nBest params: {grid.best_params_}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²:   {r2:.4f}")
    print(f"Saved model to:  {model_path}")
    print(f"Saved report to: {REPORTS_DIR / 'train_report.json'}")


def main() -> None:
    # Helpful debug (what file is running; what path did we resolve?)
    print("Running:", sys.argv[0])
    print("Resolved DATA_PATH:", DATA_PATH, "Exists?", DATA_PATH.exists())

    df, present_cats = load_data(DATA_PATH)

    # Assemble feature matrix and target vector
    X = df[FEATURE_NUMERIC + present_cats]
    y = df["Price_euros"]

    pipe = build_pipeline(FEATURE_NUMERIC, present_cats)
    grid, rmse, r2 = train_and_evaluate(pipe, X, y)
    save_artifacts(grid, rmse, r2, present_cats)


if __name__ == "__main__":
    main()
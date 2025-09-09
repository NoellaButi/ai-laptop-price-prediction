"""
test.py — Validate a previously trained model against a fresh train/test split.

What it does
------------
1) Loads the saved model from artifacts/model_gbr.pkl
2) Loads the dataset from data/raw/laptop_prices.csv (or a --data_path you pass)
3) Recreates a hold-out split and evaluates RMSE and R^2
4) Compares metrics to the baseline saved during training (reports/train_report.json)
5) Returns a non-zero exit code if performance degrades beyond a tolerance
   → useful for CI/CD or regression checks
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Project paths (robust)
# -----------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_DATA   = PROJECT_ROOT / "data" / "raw" / "laptop_prices.csv"
DEFAULT_MODEL  = PROJECT_ROOT / "artifacts" / "model_gbr.pkl"       # produced by train.py
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "train_report.json"     # produced by train.py

# Keep these in sync with train.py
FEATURE_NUMERIC = ["Ram", "Weight", "CPU_freq", "PrimaryStorage"]
FEATURE_CATS = [
    "Company","CPU_company","CPU_model","GPU_company","GPU_model","OS",
    "TypeName","Screen","PrimaryStorageType","SecondaryStorageType",
    "Touchscreen","IPSpanel","RetinaDisplay",
]
ALL_FEATURES = FEATURE_NUMERIC + FEATURE_CATS

RANDOM_STATE = 42
TEST_SIZE = 0.20


def find_data_path(cli_path: str | None) -> Path:
    """Choose dataset path in priority: CLI path > default path."""
    if cli_path:
        p = Path(cli_path)
        if p.exists():
            return p
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA
    raise FileNotFoundError(f"Dataset not found. Tried:\n - {cli_path}\n - {DEFAULT_DATA}")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the CSV and coerce numeric types; ensure all expected categorical columns exist."""
    df = pd.read_csv(path)

    # Numeric coercion (silently converts non-numeric to NaN; pipeline will impute)
    for c in FEATURE_NUMERIC + ["Price_euros"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure categorical columns exist (if a column is missing in this variant of the dataset)
    for c in FEATURE_CATS:
        if c not in df.columns:
            df[c] = None

    return df


def recreate_split(df: pd.DataFrame):
    """Create a train/test split deterministically (same random_state as training)."""
    X = df[ALL_FEATURES]
    y = df["Price_euros"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved laptop price model.")
    parser.add_argument("--data_path", help="Path to CSV (optional).")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL), help="Path to model_gbr.pkl.")
    parser.add_argument("--report_path", default=str(DEFAULT_REPORT), help="Path to train_report.json.")
    parser.add_argument("--tolerance", type=float, default=0.15,
                        help="Allowed relative RMSE degradation vs baseline (default: 0.15 = 15%).")
    args = parser.parse_args()

    # Helpful debug
    print("PY FILE __file__:", __file__)
    print("PY FILE resolved:", Path(__file__).resolve())
    print("PROJECT_ROOT:", PROJECT_ROOT)

    # Resolve inputs
    data_path = find_data_path(args.data_path)
    model_path = Path(args.model_path)
    report_path = Path(args.report_path)

    print(f"Using dataset: {data_path} (exists? {data_path.exists()})")
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run training first.", file=sys.stderr)
        sys.exit(2)

    # Load model and data
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    df = load_dataset(data_path)
    X_tr, X_te, y_tr, y_te = recreate_split(df)

    # Evaluate
    print("Evaluating on test split…")
    y_pred = model.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_te, y_pred)
    print(f"Test metrics | RMSE: {rmse:,.2f} | R²: {r2:.3f}")

    # Compare against training baseline
    if report_path.exists():
        try:
            prev = json.loads(report_path.read_text(encoding="utf-8"))
            prev_rmse = float(prev.get("rmse", rmse))
            prev_r2 = float(prev.get("r2", r2))
            print(f"Baseline (report.json) | RMSE: {prev_rmse:,.2f} | R²: {prev_r2:.3f}")

            if prev_rmse > 0:
                rel_worse = (rmse - prev_rmse) / prev_rmse
                if rel_worse > args.tolerance:
                    # Fail the process for CI/CD visibility
                    print(
                        f"RMSE degraded by {rel_worse:.1%} (> {args.tolerance:.0%} tolerance).",
                        file=sys.stderr,
                    )
                    sys.exit(3)
        except Exception as e:
            print(f"Could not read/parse {report_path}: {e}")

    # Quick smoke prediction (useful to catch schema issues quickly)
    sample = X_te.iloc[[0]].copy()
    pred_one = model.predict(sample)[0]
    print("Smoke prediction (first test row):", f"{pred_one:,.2f} EUR")

    print("Test completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
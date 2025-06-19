import joblib
import numpy as np
import pandas as pd
import os

# ✅ Load model and features
model = joblib.load(os.path.join("models", "xgboost_bike_model.pkl"))
features = joblib.load(os.path.join("models", "features.pkl"))

def make_prediction(input_df: pd.DataFrame):
    input_df = input_df[features]  # Reorder or filter columns
    preds_log = model.predict(input_df)
    preds = np.expm1(preds_log)    # Reverse log1p
    return preds

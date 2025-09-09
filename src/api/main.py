from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import joblib
import os
import io
import numpy as np
from typing import Optional


app = FastAPI(title="Insurance Fraud Prediction API")

# === CONFIG - change MODEL_FILE to switch model ===
SELECTED_MODEL_FILE = "../models/trained_models/xgboost_pipeline.pkl"  # set to other model file if needed
ARTIFACTS_DIR = "../models/artifacts"
TARGET_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "target_encoder.pkl")
CATEGORICAL_COLS_PATH = os.path.join(ARTIFACTS_DIR, "categorical_cols.pkl")
MODEL_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "model_columns.pkl")
# === end config ===
_loaded_pipeline = None
_target_encoder = None
_categorical_cols = None
_model_columns = None

def load_artifacts():
    global _loaded_pipeline, _target_encoder, _categorical_cols, _model_columns

    if not os.path.exists(SELECTED_MODEL_FILE):
        raise FileNotFoundError(f"Model pipeline file not found: {SELECTED_MODEL_FILE}")
    _loaded_pipeline = joblib.load(SELECTED_MODEL_FILE)

    if os.path.exists(TARGET_ENCODER_PATH):
        _target_encoder = joblib.load(TARGET_ENCODER_PATH)
    else:
        raise FileNotFoundError(f"Warning: Target encoder not found at {TARGET_ENCODER_PATH}")

    if os.path.exists(CATEGORICAL_COLS_PATH):
        _categorical_cols = joblib.load(CATEGORICAL_COLS_PATH)
    else:
        raise FileNotFoundError(f"Categorical columns file not found: {CATEGORICAL_COLS_PATH}")

    if os.path.exists(MODEL_COLUMNS_PATH):
        _model_columns = joblib.load(MODEL_COLUMNS_PATH)
    else:
        raise FileNotFoundError(f"Model columns file not found: {MODEL_COLUMNS_PATH}")

@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
    except Exception as e:
        # startup still continues; endpoints will raise if model missing
        print(f"Warning loading artifacts: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...), return_proba: Optional[bool] = False):
    if _loaded_pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded on server")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported")

    try:
        contents = await file.read()
        df_original = pd.read_csv(io.BytesIO(contents), na_values=["", "NA", "NaN"], keep_default_na=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV: {e}")
    
    df_for_prediction = df_original.copy()

    if _model_columns is None:
        raise HTTPException(status_code=500, detail="Model columns not loaded on server")
    try:
        df_for_prediction = df_for_prediction[_model_columns]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Uploaded CSV is missing required columns: {e}")
   
   
    if _categorical_cols is None:
        raise HTTPException(status_code=500, detail="Categorical column list not loaded")
        
    df_for_prediction = df_for_prediction.fillna(0)
    for col in _categorical_cols:
        if col in df_for_prediction.columns:
            df_for_prediction[col] = df_for_prediction[col].astype(str)

    try:
        preds = _loaded_pipeline.predict(df_for_prediction)

        confidence = None
        model_prob_for_positive = None

        if hasattr(_loaded_pipeline, "predict_proba"):
            proba_array = _loaded_pipeline.predict_proba(df_for_prediction)
            if proba_array.shape[1] == 2:
                model_prob_for_positive = proba_array[:, 1]
            try:
                pred_indices = preds.astype(int)
                confidence = proba_array[np.arange(len(preds)), pred_indices]
            except Exception:
                confidence = proba_array.max(axis=1)

        probs = None
        if return_proba and model_prob_for_positive is not None:
            probs = model_prob_for_positive

        if _target_encoder is not None:
            mapped_preds = _target_encoder.inverse_transform(preds.astype(int))
        else:
            mapped_preds = np.where(preds.astype(int) == 1, "Yes", "No")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    df_original["Model_Output"] = mapped_preds
    if probs is not None:
        df_original["Model_Prob"] = probs


    if confidence is not None:
        df_original["Confidence"] = confidence
    else:
        df_original["Confidence"] = None

    out_csv = df_original.to_csv(index=False).encode("utf-8")
    return StreamingResponse(io.BytesIO(out_csv), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=predicted_{file.filename}"})

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _loaded_pipeline is not None}
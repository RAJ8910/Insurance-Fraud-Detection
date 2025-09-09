Go in src Folder
Run API:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Run Streamlit UI (in a separate terminal):
streamlit run frontend/streamlit_app.py

Notes:
- Ensure your models/artifacts exist:
  - models/trained_models/xgboost_model.pkl
  - models/artifacts/{target_encoder.pkl,categorical_cols.pkl,model_columns.pkl,}

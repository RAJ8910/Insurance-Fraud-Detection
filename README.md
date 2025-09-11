# Insurance Fraud Detection

This project provides a machine learning solution for identifying potentially fraudulent insurance claims. It features a FastAPI backend that serves a trained XGBoost model and a Streamlit frontend for interactive predictions.

The project includes:
- **Backend**: FastAPI serving a trained XGBoost model  
- **Frontend**: Streamlit interface for interactive predictions  
- **Dataset**: [Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

---

## 📌 Tech Stack
- **Backend**: FastAPI, Uvicorn  
- **Frontend**: Streamlit  
- **Model**: XGBoost  
- **Language**: Python  
- **Notebooks**: Jupyter / Google Colab  

---

## ⚙️ Setup & Installation

> All commands should be run from the `src` directory.

### 1. Create and Activate Virtual Environment
```bash
uv venv
```

Activate the environment:  
**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
uv pip install -r requirements.txt
```

---

## 🚀 Running the Application

Make sure you are in the `src` directory with the virtual environment activated.

### 1. Start the Backend API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Launch the Frontend UI
In a separate terminal:
```bash
streamlit run frontend/streamlit_app.py
```

The Streamlit interface will be accessible at the local URL shown in the terminal.

---

## 📊 Model Retraining and EDA

The `notebooks/` directory contains Jupyter notebooks for exploratory data analysis (EDA) and model training.  
Recommended: **Google Colab** for running notebooks.

### 🔍 Exploratory Data Analysis (EDA)
- **File:** `notebooks/Insurance_Fraud_detection.ipynb`  
- **Purpose:** Data loading, cleaning, feature engineering, and analysis  
- **Output:** Generates `Data.csv` (used in training)  

### 🧠 Model Training & Retraining
- **File:** `notebooks/Model_Training_insurance.ipynb`  
- **Purpose:** Train, evaluate, and compare models using `Data.csv`  
- **Output:** Saves the best model pipeline and preprocessing artifacts (encoders, column lists, etc.) into `models/` for API usage  

---

## 📂 Project Structure
```
Insurance-Fraud-Detection/
├── models/                # Saved models and artifacts
├── notebooks/             # EDA and model training notebooks
├── src/
│   ├── api/               # FastAPI backend
│   ├── frontend/          # Streamlit frontend
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
import streamlit as st
import pandas as pd
import requests
import io

API_URL = "http://localhost:8000/predict"  

st.title("Insurance Fraud - Model Test UI")
st.markdown("Upload a CSV (use format like test/test.csv). Server returns CSV with Model_Output column.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    st.info(f"Uploading {uploaded_file.name} to API...")
    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
    params = {"return_proba": "false"}
    try:
        with st.spinner("Calling prediction API..."):
            resp = requests.post(API_URL, files=files, params=params, timeout=120)
        if resp.status_code == 200:
            annotated = pd.read_csv(io.BytesIO(resp.content))
            st.success("Prediction complete — showing results")
            st.dataframe(annotated.head(200))

            if {'PotentialFraud', 'Model_Output'}.issubset(annotated.columns):
                pf = annotated['PotentialFraud'].fillna('').astype(str).str.strip().str.lower()
                mo = annotated['Model_Output'].fillna('').astype(str).str.strip().str.lower()
                match = pf == mo
                right = int(match.sum())
                total = int(len(match))
                wrong = total - right
                accuracy = right / total if total > 0 else 0.0

                st.subheader("Prediction results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Correct", right)
                c2.metric("Incorrect", wrong)
                c3.metric("Accuracy", f"{accuracy*100:.2f}%")
                st.write(f"{right}/{total} correct")
            else:
                st.warning("Cannot compute accuracy: returned CSV missing 'Potential_fraud' and/or 'Model_Output' columns.")

            st.download_button("Download annotated CSV", resp.content, file_name=f"predicted_{uploaded_file.name}")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
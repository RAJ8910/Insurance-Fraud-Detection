import streamlit as st
import pandas as pd
import requests
import io
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Detection UI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
# This helps persist data across user interactions
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None


# --- Helper Functions ---

def create_example_dataframe():
    """Creates a sample pandas DataFrame for demonstration purposes."""
    data = {
        'feature_1': [1.2, 0.5, -0.8],
        'feature_2': [10, 25, 13],
        'feature_3': [0, 1, 0],
        'PotentialFraud': ['Yes', 'No', 'Yes'] # This column is for accuracy check
    }
    return pd.DataFrame(data)

def run_prediction(api_url: str, file_bytes: bytes, filename: str, return_proba: bool):
    """
    Sends data to the prediction API and handles the response.
    Returns the response content on success, None on failure.
    """
    files = {"file": (filename, file_bytes, "text/csv")}
    params = {"return_proba": str(return_proba).lower()}
    
    try:
        with st.spinner("🚀 Calling prediction API... This may take a moment."):
            resp = requests.post(api_url, files=files, params=params, timeout=180)
        
        if resp.status_code == 200:
            st.success("✅ Prediction successful!")
            return resp.content
        else:
            st.error(f"API Error (Status {resp.status_code}):")
            st.error(f"Details: {resp.text}")
            return None
            
    except requests.exceptions.ConnectionError as e:
        st.error("Connection Error: Could not connect to the API.")
        st.error(f"Please ensure the backend server is running at '{api_url}' and is accessible.")
        return None
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def display_results(result_bytes: bytes, original_filename: str):
    """Parses and displays the prediction results, including metrics, charts, and data."""
    try:
        annotated_df = pd.read_csv(io.BytesIO(result_bytes))
        st.session_state.prediction_results = annotated_df

        st.subheader("📊 Prediction Analysis")
        
        # Check if required columns exist for accuracy calculation
        if {'PotentialFraud', 'Model_Output'}.issubset(annotated_df.columns):
            # Normalize columns for robust comparison
            pf = annotated_df['PotentialFraud'].fillna('').astype(str).str.strip().str.lower()
            mo = annotated_df['Model_Output'].fillna('').astype(str).str.strip().str.lower()
            
            match = (pf == mo)
            right = int(match.sum())
            total = len(match)
            wrong = total - right
            accuracy = right / total if total > 0 else 0.0

            # --- Metrics and Chart ---
            col1, col2 = st.columns([1, 1])

            with col1:
                st.metric("✅ Correct Predictions", right)
                st.metric("❌ Incorrect Predictions", wrong)
                st.metric("🎯 Accuracy", f"{accuracy:.2%}")
                
            with col2:
                chart_data = pd.DataFrame({
                    'Category': ['Correct', 'Incorrect'],
                    'Count': [right, wrong],
                    'Color': ['#4CAF50', '#F44336'] # Green for correct, Red for incorrect
                })
                
                pie_chart = alt.Chart(chart_data).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(field="Category", type="nominal", scale=alt.Scale(domain=['Correct', 'Incorrect'], range=['#4CAF50', '#F44336']), legend=None),
                    tooltip=['Category', 'Count']
                ).properties(
                    title='Prediction Breakdown',
                    width=300,
                    height=300
                )
                st.altair_chart(pie_chart, use_container_width=True)

        else:
            st.warning("Cannot compute accuracy: CSV is missing 'PotentialFraud' and/or 'Model_Output' columns.")

        # --- Dataframe Expander ---
        with st.expander("📄 View Full Annotated Data", expanded=False):
            st.dataframe(annotated_df, width="stretch")
            
        # --- Download Button ---
        st.download_button(
            label="📥 Download Annotated CSV",
            data=result_bytes,
            file_name=f"predicted_{original_filename}",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error processing results: {e}")

# --- Sidebar ---
with st.sidebar:
    st.image("https://www.streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png", width=200)
    st.title("🛡️ Fraud Detection")
    st.markdown("---")
    st.header("⚙️ API Configuration")
    
    api_url = st.text_input(
        "API URL", 
        "http://localhost:8000/predict",
        help="The endpoint URL for the prediction model."
    )
    
    return_proba = st.toggle(
        "Return Probabilities", 
        value=False,
        help="If enabled, the model will return prediction probabilities along with labels."
    )
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info(
        "This application provides a user-friendly interface to test the insurance fraud detection model. "
        "Upload a CSV file or use the example data to get started."
    )

# --- Main Application ---
st.title("Insurance Fraud - Model Testing UI")
st.markdown("Upload a CSV file with provider data to get fraud predictions from the model.")

uploaded_file = st.file_uploader(
    "Choose a CSV file", 
    type=["csv"],
    help="The CSV should be in the same format as the training data."
)

if uploaded_file is not None:
    # Clear previous results when a new file is uploaded
    st.session_state.prediction_results = None

    st.info(f"File '{uploaded_file.name}' selected. Click below to run prediction.")

    if st.button("Run Prediction", key="run_upload"):
        file_bytes = uploaded_file.getvalue()
        result_content = run_prediction(api_url, file_bytes, uploaded_file.name, return_proba)

        if result_content:
            display_results(result_content, uploaded_file.name)

# A placeholder to show a message if no action has been taken yet
if uploaded_file is None and st.session_state.prediction_results is None:
     st.info("Please upload a file or use the example data to start.")

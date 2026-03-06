import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Explainable AI", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/churn_rf_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/client_data.csv")

model_artifacts = load_model()
model = model_artifacts['model']
expected_features = model_artifacts['features']
df = load_data()

st.title("Explainable AI & Action Alerts")
st.markdown("Understand **why** the model predicts churn, and review automated intervention recommendations.")
st.divider()
analysis_mode = st.radio(
    "Select Analysis Mode:",
    ["Analyze Custom Scenario (From Simulator)", "Analyze Existing Client Database"],
    horizontal=True
)

client_features = None
client_data_raw = None

if analysis_mode == "Analyze Custom Scenario (From Simulator)":
    if 'custom_client_data' in st.session_state:
        client_features = st.session_state['custom_client_data']
        client_data_raw = st.session_state['custom_raw_data']
        st.info("Currently analyzing the custom parameters you set in the Scenario Simulator.")
    else:
        st.warning("No custom scenario found! Go to the 'Scenario Simulator' page, adjust the sliders, and click the Generate button first.")

elif analysis_mode == "Analyze Existing Client Database":
    high_risk_clients = df[df['Churn'] == 1]['Company_ID'].tolist()
    selected_client = st.selectbox("Select an At-Risk Client Account:", options=high_risk_clients[:50])
    
    if selected_client:
        client_data_raw = df[df['Company_ID'] == selected_client].iloc[0]
        client_features = pd.DataFrame([client_data_raw])
        client_features = client_features.drop(columns=['Company_ID', 'Churn'])
        client_features = pd.get_dummies(client_features)
        
        for col in expected_features:
            if col not in client_features.columns:
                client_features[col] = 0
        client_features = client_features[expected_features]
if client_features is not None:
    churn_prob = model.predict_proba(client_features)[0][1] * 100
    st.markdown(f"### Current Risk Score: **{churn_prob:.1f}%**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Reasoning (SHAP Waterfall)")
        st.markdown("Red bars push the risk *higher*, blue bars push the risk *lower*.")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(client_features)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0, :, 1], show=False) 
        st.pyplot(fig)
        
    with col2:
        st.subheader("Action Alerts")
        
        if client_data_raw['Days_Since_Last_Login'] > 30:
            st.warning("⚠️ **Low Engagement:** Client hasn't logged in recently. Trigger re-engagement workflow.")
        
        if client_data_raw['Support_Tickets_Last_Month'] > 5:
            st.error("⚠️ **High Friction:** Elevated support tickets. Assign a CSM immediately.")
            
        if client_data_raw['Contract_Type'] == 'Month-to-Month':
            st.success("💡 **Contract Opportunity:** Offer a discount to lock into an annual contract.")
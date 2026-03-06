import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Scenario Simulator", layout="wide")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "models/churn_rf_model.pkl")
    return joblib.load(model_path)

model_artifacts = load_model()
model = model_artifacts['model']
expected_features = model_artifacts['features']

st.title("Risk Intervention Simulator")
st.markdown("Adjust client parameters below to simulate how different interventions (e.g., upgrading a contract, increasing engagement) impact the probability of churn in real-time.")
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Client Engagement Metrics")
    login_days = st.slider("Days Since Last Login", min_value=1, max_value=90, value=15, 
                           help="How long since the client last used the platform?")
    tickets = st.slider("Support Tickets (Last Month)", min_value=0, max_value=15, value=2,
                        help="Number of active issues reported.")
    users = st.number_input("Active Users on Account", min_value=5, max_value=500, value=50)

with col2:
    st.subheader("Account & Financials")
    contract = st.selectbox("Contract Type", options=['Month-to-Month', 'One Year', 'Two Year'])
    mrr = st.number_input("Monthly Recurring Revenue ($)", min_value=500, max_value=10000, value=2500)
    age = st.number_input("Account Age (Months)", min_value=1, max_value=60, value=12)

st.divider()
input_dict = {feat: 0 for feat in expected_features}
input_dict['Account_Age_Months'] = age
input_dict['Monthly_Recurring_Revenue'] = mrr
input_dict['Active_Users'] = users
input_dict['Support_Tickets_Last_Month'] = tickets
input_dict['Days_Since_Last_Login'] = login_days

if contract == 'One Year':
    if 'Contract_Type_One Year' in input_dict:
        input_dict['Contract_Type_One Year'] = 1
elif contract == 'Two Year':
    if 'Contract_Type_Two Year' in input_dict:
        input_dict['Contract_Type_Two Year'] = 1

input_df = pd.DataFrame([input_dict])
churn_probability = model.predict_proba(input_df)[0][1] * 100
st.subheader("Predicted Churn Risk")

if churn_probability < 30:
    st.success(f"Low Risk: {churn_probability:.1f}% probability of churn.")
    st.progress(int(churn_probability))
elif churn_probability < 60:
    st.warning(f"Moderate Risk: {churn_probability:.1f}% probability of churn.")
    st.progress(int(churn_probability))
else:
    st.error(f"High Risk: {churn_probability:.1f}% probability of churn. Immediate intervention required.")
    st.progress(int(churn_probability))

st.divider()
st.subheader("Deep Dive Analysis")
st.markdown("Want to see the mathematical breakdown of this specific scenario?")

if st.button("Generate SHAP Explanation for this Scenario"):
    st.session_state['custom_client_data'] = input_df
    st.session_state['custom_raw_data'] = client_data_raw = {
        'Days_Since_Last_Login': login_days,
        'Support_Tickets_Last_Month': tickets,
        'Contract_Type': contract
    }
    st.success("Scenario saved! Click 'Explainable AI' in the sidebar to view the breakdown.")

import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Churn Command Center", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "data/client_data.csv")
    return pd.read_csv(data_path)

df = load_data()

st.title("Executive Command Center")
st.markdown("### B2B SaaS Churn & Revenue Analytics")
st.markdown("This dashboard identifies at-risk accounts to optimize retention strategies and protect recurring revenue.")
st.divider()

total_clients = len(df)
churned_clients = len(df[df['Churn'] == 1])
churn_rate = (churned_clients / total_clients) * 100

revenue_at_risk = df[df['Churn'] == 1]['Monthly_Recurring_Revenue'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Active Accounts", f"{total_clients:,}")
col2.metric("Historical Churn Rate", f"{churn_rate:.1f}%")
col3.metric("MRR at Risk", f"${revenue_at_risk:,.2f}")
st.divider()

st.subheader("Risk Distribution Analysis")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_contract = px.histogram(
        df, x="Contract_Type", color="Churn",
        barmode="group",
        title="Churn Volume by Contract Type",
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        labels={"Churn": "Churn Status (1=Yes)"}
    )
    st.plotly_chart(fig_contract, use_container_width=True)

with chart_col2:
    fig_scatter = px.scatter(
        df, x="Days_Since_Last_Login", y="Support_Tickets_Last_Month", 
        color="Churn", opacity=0.6,
        title="Engagement: Logins vs Support Tickets",
        color_continuous_scale=["#2ECC71", "#E74C3C"]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

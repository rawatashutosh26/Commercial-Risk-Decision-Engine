# Commercial Risk Decision Engine & Churn Prediction Lakehouse

**Live Dashboard:** [Click Here](https://commercial-risk-decision-engine-cappkgmfjfjssdpks3xrodu.streamlit.app/)

## 📌 Project Overview
An end-to-end **Data Engineering & Decision Analytics** platform designed to identify high-value customers, predict churn risks, and recommend data-driven retention strategies. 

This solution bridges the gap between heavy data infrastructure and business intelligence. It is divided into two phases:
1. **The Backend (Azure Databricks):** Processes raw transaction logs into a Medallion Lakehouse, engineering a "Customer 360" feature store using RFM Analysis.
2. **The Frontend (Streamlit):** A deployed, interactive web application allowing stakeholders to simulate "What-If" intervention scenarios and understand model logic via Explainable AI.

## 🏗 Architecture & Tech Stack
* **Cloud & Data Engineering:** Azure Databricks, Apache Spark (PySpark), Delta Lake (Medallion Architecture).
* **Machine Learning:** Spark MLlib, Scikit-Learn, Random Forest, Logistic Regression.
* **Frontend Analytics & Explainability:** Streamlit, SHAP (SHapley Additive exPlanations), Plotly.
* **Visualization:** Microsoft Power BI.
* **Languages:** Python, SQL, DAX.

---

## ⚙️ Phase 1: Data Engineering Pipeline (Databricks)
### 1. Ingestion (Bronze Layer)
* Ingested raw CSV transaction data from the Online Retail dataset.
* Implemented **Schema Enforcement** to handle data type validation at the source.
* Archived raw history in **Delta Tables**.

### 2. Transformation (Silver Layer)
* Performed data cleaning: Handling null CustomerIDs and formatting timestamps.
* Deduplicated records to ensure data integrity.
* Used **Delta Lake MERGE** logic (simulated) for upserts.

### 3. Feature Engineering (Gold Layer)
* Aggregated transaction logs into a **Customer 360 Profile**.
* Calculated **RFM Metrics**:
    * **Recency:** Days since last purchase.
    * **Frequency:** Total count of orders.
    * **Monetary:** Total lifetime spend.
* Created target labels (`is_high_value`) for the ML model.

### 4. Machine Learning & BI
* Trained a **Spark ML Logistic Regression** model to classify customers, achieving **95% Accuracy**.
* Visualized "At-Risk VIPs" in Power BI to drive retention campaigns.
  <img width="880" height="245" alt="Screenshot 2025-12-13 025332" src="https://github.com/user-attachments/assets/604164b8-f404-42ff-8451-7c69ee421166" />

### Power BI Dashboard Preview
<img width="1298" height="725" alt="Screenshot 2025-12-13 025123" src="https://github.com/user-attachments/assets/b35008b0-7c72-4816-8949-2889926cb1ec" />
<img width="1301" height="673" alt="Screenshot 2025-12-13 024155" src="https://github.com/user-attachments/assets/7835f863-1ad5-4709-af38-b814fa694282" />

---

## 💻 Phase 2: Decision Analytics Web App (Streamlit)
To make the predictive models actionable for business stakeholders, a lightweight version of the model was deployed into a fully interactive web application.

### 1. Executive Command Center
* Provides real-time tracking of active accounts, historical churn rates, and total **Revenue at Risk (MRR)**.
* Uses Plotly to visualize churn distribution across contract types and engagement metrics.

### 2. "What-If" Scenario Simulator
* Transforms a static prediction into a dynamic decision engine. 
* Users can adjust client parameters (e.g., reducing support tickets, upgrading contract terms) to see how interventions impact the probability of churn in real-time.

### 3. Explainable AI & Action Alerts (SHAP)
* Integrates `shap` TreeExplainer to break down the mathematical "why" behind individual client predictions.
* Generates dynamic Waterfall charts showing which specific features are driving risk up (red) or down (blue).
* Translates mathematical outputs into automated, plain-text consulting recommendations (e.g., "Elevated support tickets detected. Assign CSM immediately.").

### Streamlit Dashboard Preview
<img width="1564" height="831" alt="Screenshot 2026-03-06 170504" src="https://github.com/user-attachments/assets/8e9ccc3f-0df0-4c3e-aff1-7d9fba8ff5ba" />
<img width="1575" height="872" alt="Screenshot 2026-03-06 170343" src="https://github.com/user-attachments/assets/25abf4e1-f29c-400b-818d-7db299c73cd3" />
<img width="1551" height="868" alt="Screenshot 2026-03-06 170406" src="https://github.com/user-attachments/assets/30c5abbd-6634-4dcf-998f-2e228447f775" />
<img width="1549" height="806" alt="Screenshot 2026-03-06 170421" src="https://github.com/user-attachments/assets/68c7e2c0-d1ab-4211-9614-67a3734f8d9b" />


---

## 🚀 How to Run Locally

**For the Databricks Pipeline:**
1. Upload the `.py` notebooks to your Databricks Workspace.
2. Run the pipeline `01` through `04`.
3. Open the `.pbix` file in Power BI Desktop to view the static dashboard.

**For the Streamlit Web App:**
1. Clone this repository.
2. Install the required Python libraries:
   ```bash
   pip install -r 2_interactive_dashboard/requirements.txt
   ```
3. Navigate to the dashboard directory and launch the app:
   ```bash
   cd 2_dashboard
   streamlit run Home.py
   ```

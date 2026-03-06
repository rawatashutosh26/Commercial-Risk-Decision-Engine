# 🛒 Commercial Risk Decision Engine & Churn Prediction Lakehouse

## 📌 Project Overview
A full-lifecycle **Data Engineering & Machine Learning** project designed to identify high-value customers and predict churn risks. 

Built on **Azure Databricks**, this solution processes raw ERP transaction logs, builds a "Customer 360" feature store using **RFM Analysis** (Recency, Frequency, Monetary), and deploys a **Logistic Regression Model** to predict future purchasing behavior. The final insights are delivered via an interactive **Power BI** dashboard.

## 🏗 Architecture & Tech Stack
* **Platform:** Azure Databricks (Community Edition)
* **Storage:** Delta Lake (Medallion Architecture: Bronze $\to$ Silver $\to$ Gold)
* **Compute:** Apache Spark (PySpark)
* **Machine Learning:** Spark MLlib (Logistic Regression)
* **Visualization:** Microsoft Power BI
* **Language:** Python, SQL, DAX

## 🔧 Pipeline Breakdown
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
* Trained a **Spark ML Logistic Regression** model to classify customers.
* Achieved **95% Accuracy** on test data.
* Visualized "At-Risk VIPs" in Power BI to drive retention campaigns.
  <img width="880" height="245" alt="Screenshot 2025-12-13 025332" src="https://github.com/user-attachments/assets/604164b8-f404-42ff-8451-7c69ee421166" />


## 📊 Dashboard Preview
<img width="1298" height="725" alt="Screenshot 2025-12-13 025123" src="https://github.com/user-attachments/assets/b35008b0-7c72-4816-8949-2889926cb1ec" />

<img width="1301" height="673" alt="Screenshot 2025-12-13 024155" src="https://github.com/user-attachments/assets/7835f863-1ad5-4709-af38-b814fa694282" />



## 🚀 How to Run
1.  Clone the repo.
2.  Upload the `.py` notebooks to your Databricks Workspace.
3.  Run the pipeline `01` through `04`.
4.  Open the `.pbix` file in Power BI Desktop to view the dashboard.


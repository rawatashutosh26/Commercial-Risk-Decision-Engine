# Databricks notebook source
# Next Layer -> GOLD LAYER (Feature Engineering / RFM)

from pyspark.sql.functions import max, sum, count, datediff, current_date, expr, lit, when, col

print("Building Customer 360 Profile...")
df_silver = spark.read.table("silver_retail_clean")

# 1. Calculate RFM Metrics
# Recency - How many days since last purchase?
# Frequency - How many purchases total?
# Monetary - How much money spent total?

# Set a reference date (simulating "today" as the last day in dataset)
max_date = df_silver.select(max("InvoiceDate")).collect()[0][0]

customer_360 = df_silver.groupBy("CustomerID").agg(
    # For Recency
    datediff(lit(max_date), max("InvoiceDate")).alias("recency_days"),
    # For Frequency
    count("InvoiceNo").alias("frequency_purchase_count"),
    # For Monetary
    sum("TotalAmount").alias("monetary_total_spend"),
    # For Extra Features
    count(when(col("Description").contains("GIFT"), True)).alias("gift_purchases")
)

# 2. Define "Target" Label for ML
# Let's pretend 'High Value' means they spent > $2000
customer_features = customer_360.withColumn(
    "is_high_value", 
    when(col("monetary_total_spend") > 2000, 1).otherwise(0)
)

# 3. Save as Gold Table
customer_features.write.format("delta").mode("overwrite").saveAsTable("gold_customer_features")

print("Gold Layer Ready.")
display(customer_features)

# COMMAND ----------


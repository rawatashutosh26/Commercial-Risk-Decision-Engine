# Databricks notebook source
# Next Layer -> SILVER LAYER (Data Quality & Standardization)

from pyspark.sql.functions import to_timestamp, col, count, when

print("Loading from Bronze...")
df_bronze = spark.read.table("bronze_retail_trx")

# 1. Data Quality Check -> Check for null CustomerIDs before we process
null_count = df_bronze.filter(col("CustomerID").isNull()).count()
print(f"Found {null_count} transactions with no CustomerID ---> Dropping them.")

# 2. Transformation Logic
df_silver = df_bronze \
    .filter(col("CustomerID").isNotNull()) \
    .filter(col("Quantity") > 0) \
    .withColumn("InvoiceDate", to_timestamp(col("InvoiceDate"), "M/d/yy H:mm")) \
    .withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))

# 3. Deduplication -> To ensure uniqueness
df_silver = df_silver.dropDuplicates(["InvoiceNo", "StockCode", "CustomerID"])

# 4. Write to Silver
print(f"Saving {df_silver.count()} clean records to Silver...")
df_silver.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable("silver_retail_clean")

print("Silver Layer Complete.")

# COMMAND ----------


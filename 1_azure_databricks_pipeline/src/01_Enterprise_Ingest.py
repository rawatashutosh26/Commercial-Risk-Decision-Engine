# Databricks notebook source
# First layer -> BRONZE LAYER (Ingestion & Archiving)

from pyspark.sql.functions import current_timestamp, to_date, col, lit
from pyspark.sql.types import *

# 1. We will define the schema for the raw data
df = StructType([
    StructField("InvoiceNo", StringType(), True),
    StructField("StockCode", StringType(), True),
    StructField("Description", StringType(), True),
    StructField("Quantity", IntegerType(), True),
    StructField("InvoiceDate", StringType(), True), 
    StructField("UnitPrice", DoubleType(), True),
    StructField("CustomerID", DoubleType(), True),
    StructField("Country", StringType(), True)
])

# 2. Read Raw Data as a Spark DataFrame
print("Ingesting raw retail data...")
raw_df = spark.read.format("csv") \
  .option("header", "true") \
  .schema(df) \
  .load("/databricks-datasets/online_retail/data-001/data.csv")

# 3. Add Metadata
bronze_df = raw_df.withColumn("ingestion_timestamp", current_timestamp()) \
                  .withColumn("source_system", lit("ERP_SAP_V1"))

# 4. Write to Managed Delta Table (Bronze)
print(f"Saving {bronze_df.count()} raw records to Bronze reatil file...")
bronze_df.write.format("delta").mode("overwrite").saveAsTable("bronze_retail_trx")

print("Successfully ingested raw data to-> Bronze Layer Complete.")

# COMMAND ----------


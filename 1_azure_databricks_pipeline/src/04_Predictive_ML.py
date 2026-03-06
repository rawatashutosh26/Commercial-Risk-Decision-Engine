# Databricks notebook source
# COMMAND ----------
# Final Layer -> PREDICTIVE ANALYTICS (Spark ML)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

print("Initializing Machine Learning Pipeline...")
data = spark.read.table("gold_customer_features")

# 1. Prepare Features for ML (Vectorization) -> Here we use Recency and Frequency to predict if they are 'High Value' (Monetary is cheating, so we exclude it)
assembler = VectorAssembler(
    inputCols=["recency_days", "frequency_purchase_count", "gift_purchases"],
    outputCol="features"
)

data_vec = assembler.transform(data)

# 2. Split Data (Training vs Testing)
train_data, test_data = data_vec.randomSplit([0.7, 0.3])

# 3. Train Model (Logistic Regression)
lr = LogisticRegression(labelCol="is_high_value", featuresCol="features")
print("Training model...")
model = lr.fit(train_data)

# 4. Evaluate Model
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="is_high_value", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy*100:.2f}%")

# 5. Visualize Results
display(predictions.select("CustomerID", "prediction", "is_high_value", "probability"))

final_data = predictions.join(data, "CustomerID").drop("features", "rawFeatures")
display(final_data)

# COMMAND ----------


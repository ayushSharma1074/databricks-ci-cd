# Databricks notebook source
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
model_full_name = dbutils.widgets.get("model_name")

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from sklearn.metrics import accuracy_score, f1_score

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Read test set written by train task
test_df = spark.read.table(f"{catalog_name}.{schema_name}.wine_test").toPandas()
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_full_name}@Champion")
y_pred = loaded_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

with mlflow.start_run():
    mlflow.log_metric("eval_accuracy", accuracy)
    mlflow.log_metric("eval_f1_score", f1)

print(f"Evaluation — Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

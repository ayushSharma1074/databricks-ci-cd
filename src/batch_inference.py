# Databricks notebook source
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
model_full_name = dbutils.widgets.get("model_name")

# COMMAND ----------

import mlflow
import mlflow.pyfunc

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Read input features — labels dropped to simulate production inference data
inference_df = (
    spark.read.table(f"{catalog_name}.{schema_name}.wine_test")
    .toPandas()
    .drop(columns=["label"])
)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_full_name}@Champion")
inference_df["prediction"] = loaded_model.predict(inference_df)

# COMMAND ----------

spark.createDataFrame(inference_df).write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.wine_predictions"
)

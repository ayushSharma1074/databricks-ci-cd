# Databricks notebook source
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
model_full_name = dbutils.widgets.get("model_name")

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="label")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

with mlflow.start_run() as run:
    n_estimators = 100
    max_depth = 5

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("dataset", "sklearn_wine")

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))

    signature = infer_signature(X_test, y_pred)
    input_example = X_test[:5]
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

    run_id = run.info.run_id

# COMMAND ----------

model_version = mlflow.register_model(f"runs:/{run_id}/model", model_full_name)

client = MlflowClient()
client.set_registered_model_alias(model_full_name, "Champion", model_version.version)

# COMMAND ----------

# Write test set to Delta for the evaluate task to consume
test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
spark.createDataFrame(test_df).write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.wine_test"
)

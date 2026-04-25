FROM apache/airflow:2.8.3

# Switch to root to install OS-level dependencies
USER root
RUN apt-get update && apt-get install -y libgomp1

# Switch back to the airflow user to install Python packages
USER airflow
RUN pip install --no-cache-dir pandas scikit-learn xgboost shap psycopg2-binary kaggle lightgbm
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import logging
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import pickle
import os
import shap
from airflow.models import TaskInstance

# --- PHASE 1 & 2: DATA ENGINEERING FUNCTIONS ---

def load_local_data():
    """Reads the CSV from the local mounted volume and prints basic stats."""
    filepath = '/opt/airflow/data/raw_mental_health_risk_data.csv'
    logging.info(f"Attempting to read file at: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logging.info("SUCCESS! Data loaded into Pandas dataframe.")
        logging.info(f"Dataset Shape (Rows, Columns): {df.shape}")
        logging.info(f"Columns found: {df.columns.tolist()}")
    except FileNotFoundError:
        logging.error(f"Could not find the file at {filepath}. Please check the data folder!")
        raise

def clean_and_preprocess_data():
    """STRICTLY cleans the raw data by dropping missing values."""
    raw_path = '/opt/airflow/data/raw_mental_health_risk_data.csv'
    clean_path = '/opt/airflow/data/clean_mental_health_data.csv'
    
    logging.info("Starting data cleaning...")
    df = pd.read_csv(raw_path)
    
    initial_shape = df.shape
    df = df.dropna()
    
    logging.info(f"Dropped missing data. Shape changed from {initial_shape} to {df.shape}.")
    
    df.to_csv(clean_path, index=False)
    logging.info(f"Cleaned data saved to {clean_path}.")

def execute_feature_engineering():
    """Creates composite features and splits data into BI and ML formats."""
    clean_path = '/opt/airflow/data/clean_mental_health_data.csv'
    bi_path = '/opt/airflow/data/dashboard_ready_data.csv'
    ml_path = '/opt/airflow/data/ml_ready_data.csv'
    
    logging.info("Starting feature engineering...")
    df = pd.read_csv(clean_path)
    
    # 1. Domain Engineering (Composite Scores)
    df['composite_lifestyle_score'] = (df['sleep_hours'] / 10) + (df['physical_activity_hours_per_week'] / 14)
    
    # 2. Save the "BI/Dashboard Ready" dataset
    df.to_csv(bi_path, index=False)
    logging.info("Dashboard-ready data saved for Postgres.")
    
    # 3. ML Engineering (Categorical Encoding)
    categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 4. Save the "ML Ready" dataset
    df_encoded.to_csv(ml_path, index=False)
    logging.info("ML-ready data saved for model training.")

def load_to_postgres():
    """Loads the dashboard-ready data into the PostgreSQL Data Warehouse."""
    bi_path = '/opt/airflow/data/dashboard_ready_data.csv'
    df = pd.read_csv(bi_path)
    
    db_url = 'postgresql+psycopg2://airflow:airflow@postgres:5432/airflow'
    engine = create_engine(db_url)
    
    logging.info("Connected to PostgreSQL. Building tables...")
    
    dim_demographics = df[['age', 'gender', 'marital_status', 'education_level', 'employment_status']].drop_duplicates().reset_index(drop=True)
    dim_demographics['demo_id'] = dim_demographics.index + 1
    dim_demographics.to_sql('dim_demographics', engine, if_exists='replace', index=False)
    
    df.to_sql('staging_mental_health_records', engine, if_exists='replace', index=False)
    logging.info("staging_mental_health_records table created and fully loaded!")

# --- PHASE 3: MACHINE LEARNING FUNCTIONS ---

def prepare_data_for_ml():
    """Loads the pre-encoded, pre-engineered ML data and splits it."""
    ml_path = '/opt/airflow/data/ml_ready_data.csv'
    df_encoded = pd.read_csv(ml_path)
    
    X = df_encoded.drop('mental_health_risk', axis=1)
    y = df_encoded['mental_health_risk']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest():
    logging.info("Training Random Forest Baseline...")
    X_train, X_test, y_train, y_test = prepare_data_for_ml()
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    score = f1_score(y_test, preds, average='weighted')
    logging.info(f"Random Forest Weighted F1-Score: {score:.4f}")
    
    with open('/opt/airflow/models/random_forest.pkl', 'wb') as f:
        pickle.dump(rf, f)

def train_xgboost():
    logging.info("Training XGBoost Model...")
    X_train, X_test, y_train, y_test = prepare_data_for_ml()
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    
    preds = xgb.predict(X_test)
    score = f1_score(y_test, preds, average='weighted')
    logging.info(f"XGBoost Weighted F1-Score: {score:.4f}")
    
    with open('/opt/airflow/models/xgboost.pkl', 'wb') as f:
        pickle.dump(xgb, f)

def train_logistic_regression():
    logging.info("Training Logistic Regression Model with embedded Scaler...")
    X_train, X_test, y_train, y_test = prepare_data_for_ml()
    
    # Pack the scaler and the model into a single pipeline object
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    lr_pipeline.fit(X_train, y_train)
    
    preds = lr_pipeline.predict(X_test)
    score = f1_score(y_test, preds, average='weighted')
    logging.info(f"Logistic Regression Weighted F1-Score: {score:.4f}")
    
    with open('/opt/airflow/models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_pipeline, f)

def train_lightgbm():
    logging.info("Training LightGBM Model...")
    X_train, X_test, y_train, y_test = prepare_data_for_ml()
    
    lgbm = LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)
    
    preds = lgbm.predict(X_test)
    score = f1_score(y_test, preds, average='weighted')
    logging.info(f"LightGBM Weighted F1-Score: {score:.4f}")
    
    with open('/opt/airflow/models/lightgbm.pkl', 'wb') as f:
        pickle.dump(lgbm, f)

def evaluate_and_select_champion():
    """Evaluates all trained models and promotes the best one to Champion."""
    logging.info("Evaluating all models to select the champion...")
    X_train, X_test, y_train, y_test = prepare_data_for_ml()
    
    models_to_test = ['random_forest', 'xgboost', 'logistic_regression', 'lightgbm']
    best_score = 0
    champion_name = ""
    
    for model_name in models_to_test:
        try:
            with open(f'/opt/airflow/models/{model_name}.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Because LR is now a pipeline, don't need manual scaling here
            preds = model.predict(X_test)
                
            score = f1_score(y_test, preds, average='weighted')
            logging.info(f"{model_name.upper()} Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                champion_name = model_name
                
        except Exception as e:
            logging.warning(f"Could not evaluate {model_name}: {e}")
            
    logging.info(f"🏆 CHAMPION MODEL SELECTED: {champion_name.upper()} with F1-Score: {best_score:.4f}")
    
    with open(f'/opt/airflow/models/{champion_name}.pkl', 'rb') as f:
        champion_model = pickle.load(f)
    with open('/opt/airflow/models/champion_model.pkl', 'wb') as f:
        pickle.dump(champion_model, f)

# --- PHASE 4: EXPLAINABILITY & RECOMMENDATIONS ---

def generate_shap_values():
    """Uses SHAP to explain the Champion Model's decision-making process."""
    logging.info("Generating SHAP explainability for the Champion Model...")
    X_train, X_test, y_train, y_test = prepare_data_for_ml()
    
    with open('/opt/airflow/models/champion_model.pkl', 'rb') as f:
        champion_model = pickle.load(f)
        
    X_sample = X_test.sample(100, random_state=42)
    
    # Extract the actual model if it's wrapped in a Pipeline
    if isinstance(champion_model, Pipeline):
        model_to_explain = champion_model.named_steps['classifier']
        # Scale the SHAP sample if the model expects scaled data
        X_sample_processed = pd.DataFrame(
            champion_model.named_steps['scaler'].transform(X_sample), 
            columns=X_sample.columns
        )
    else:
        model_to_explain = champion_model
        X_sample_processed = X_sample

    explainer = shap.Explainer(model_to_explain)
    shap_values = explainer(X_sample_processed)
    
    logging.info(f"SHAP values successfully calculated for shape: {shap_values.shape}")
    logging.info("Ready to pass explainability metrics to the recommendation engine!")

def build_actionable_recommendations():
    """Generates predictions and maps them to actionable advice."""
    logging.info("Generating predictions and actionable recommendations...")
    
    bi_path = '/opt/airflow/data/dashboard_ready_data.csv'
    df_bi = pd.read_csv(bi_path)
    
    ml_path = '/opt/airflow/data/ml_ready_data.csv'
    df_ml = pd.read_csv(ml_path)
    X = df_ml.drop('mental_health_risk', axis=1)
    
    with open('/opt/airflow/models/champion_model.pkl', 'rb') as f:
        champion = pickle.load(f)
        
    # The pipeline automatically handles scaling if Logistic Regression won
    df_bi['predicted_risk'] = champion.predict(X)
    
    def get_recommendation(row):
        if row['predicted_risk'] == 0:
            return "Low Risk: Maintain current healthy lifestyle habits."
        elif row['predicted_risk'] == 1:
            if row['work_stress_level'] > 7: 
                return "Moderate Risk: High work stress detected. Consider discussing workload with HR or taking short breaks."
            elif row['sleep_hours'] < 6: 
                return "Moderate Risk: Sleep deprivation detected. Prioritize getting 7-8 hours of sleep."
            else:
                return "Moderate Risk: Monitor overall stress levels and practice daily self-care."
        else:
            return "High Risk: Significant risk indicators detected. Please consult a mental health professional."

    df_bi['actionable_recommendation'] = df_bi.apply(get_recommendation, axis=1)
    
    out_path = '/opt/airflow/data/final_predictions_with_recs.csv'
    df_bi.to_csv(out_path, index=False)
    logging.info(f"SUCCESS! Recommendations generated and saved to {out_path}")

def update_postgres_with_predictions_and_recs():
    """Loads the final predictions and recommendations into Postgres for the Dashboard."""
    out_path = '/opt/airflow/data/final_predictions_with_recs.csv'
    df = pd.read_csv(out_path)
    
    db_url = 'postgresql+psycopg2://airflow:airflow@postgres:5432/airflow'
    engine = create_engine(db_url)
    
    logging.info("Pushing final results to PostgreSQL Data Warehouse...")
    
    df.to_sql('mental_health_predictions', engine, if_exists='replace', index=False)
    logging.info("SUCCESS! 'mental_health_predictions' table is live and ready for the dashboard.")

def export_runtime_report(**context):
    """Exports task execution times to a CSV report for documentation."""
    from airflow.models import DagRun, TaskInstance
    from airflow.utils.db import create_session
    
    try:
        dag_id = 'mental_health_ml_pipeline'
        execution_date = context['execution_date']
        
        with create_session() as session:
            task_instances = session.query(TaskInstance).filter(
                TaskInstance.dag_id == dag_id,
                TaskInstance.execution_date == execution_date
            ).all()
        
        # Build report
        report_data = []
        for ti in task_instances:
            if ti.start_date and ti.end_date:
                duration = (ti.end_date - ti.start_date).total_seconds()
                report_data.append({
                    'task_id': ti.task_id,
                    'start_time': ti.start_date,
                    'end_time': ti.end_date,
                    'duration_seconds': round(duration, 2),
                    'state': ti.state
                })
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            report_path = '/opt/airflow/data/pipeline_runtime_report.csv'
            report_df.to_csv(report_path, index=False)
            logging.info(f"✅ Runtime report saved to {report_path}")
            logging.info(f"\n{report_df.to_string()}")
    except Exception as e:
        logging.warning(f"Could not generate runtime report: {e}")

# --- DAG DEFINITION ---

default_args = {
    'owner': 'group_18',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mental_health_ml_pipeline',
    default_args=default_args,
    description='IS3107 End-to-End Pipeline with ML and SHAP',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['is3107', 'ml', 'postgres'],
) as dag:

    # --- Phase 1 & 2: Data Ingestion and Transformation ---
    start_pipeline = EmptyOperator(task_id='start_pipeline')
    
    ingest_raw_data = PythonOperator(
        task_id='ingest_raw_data_local',
        python_callable=load_local_data
    )
    
    clean_and_preprocess = PythonOperator(
        task_id='clean_and_preprocess_data',
        python_callable=clean_and_preprocess_data
    )
    
    engineer_features = PythonOperator(
        task_id='engineer_features',
        python_callable=execute_feature_engineering
    )

    load_to_postgres_task = PythonOperator(
        task_id='load_to_postgres_star_schema',
        python_callable=load_to_postgres
    )

    # --- Phase 3: Parallel Model Training ---
    train_rf = PythonOperator(
        task_id='train_random_forest_model',
        python_callable=train_random_forest
    )
    
    train_xgb = PythonOperator(
        task_id='train_xgboost_model',
        python_callable=train_xgboost
    )
    
    train_lr = PythonOperator(
        task_id='train_logistic_regression_model',
        python_callable=train_logistic_regression
    )
    
    train_lgbm = PythonOperator(
        task_id='train_lightgbm_model',
        python_callable=train_lightgbm
    )

    # --- Phase 4: Explainability & Recommendations ---
    evaluate_and_select = PythonOperator(
        task_id='evaluate_models_and_select_champion',
        python_callable=evaluate_and_select_champion
    )
    
    generate_shap = PythonOperator(
        task_id='generate_shap_values',
        python_callable=generate_shap_values
    )
    
    build_recs = PythonOperator(
        task_id='build_actionable_recommendations',
        python_callable=build_actionable_recommendations
    )
    
    update_db = PythonOperator(
        task_id='update_postgres_with_predictions_and_recs',
        python_callable=update_postgres_with_predictions_and_recs
    )
    
    export_runtime = PythonOperator(
        task_id='export_runtime_report',
        python_callable=export_runtime_report,
        provide_context=True
    )

    end_pipeline = EmptyOperator(task_id='end_pipeline')

    # ==========================================
    # SETTING UP THE DEPENDENCIES (The Flow)
    # ==========================================
    
    start_pipeline >> ingest_raw_data >> clean_and_preprocess >> engineer_features >> load_to_postgres_task
    
    load_to_postgres_task >> [train_rf, train_xgb, train_lr, train_lgbm]
    
    [train_rf, train_xgb, train_lr, train_lgbm] >> evaluate_and_select
    
    evaluate_and_select >> generate_shap >> build_recs >> update_db >> export_runtime >> end_pipeline
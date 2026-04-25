from setuptools import setup, find_packages

setup(
    name="mental-health-project",
    version="1.0.0",
    description="Mental Health Risk Prediction Dashboard with Airflow Pipeline",
    author="Your Name",
    python_requires=">=3.12.3",  # Enforce Python 3.12.3 or higher
    packages=find_packages(),
    install_requires=[
        "apache-airflow==2.11.2",
        "pandas==3.0.2",
        "scikit-learn==1.8.0",
        "xgboost==3.2.0",
        "shap==0.51.0",
        "streamlit==1.56.0",
        "plotly==5.24.0",
        "psycopg2-binary==2.9.9",
        "kaggle==1.6.14",
        "numpy==2.1.3",
        "matplotlib==3.10.0",
        "seaborn==0.13.2",
        "sqlalchemy==1.4.52",
        "lightgbm==4.6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

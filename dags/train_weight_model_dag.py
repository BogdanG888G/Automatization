from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

def train_weight_model():
    INPUT_CSV = "ml_models/product_enrichment/labeled_weight_products.csv"
    MODEL_PATH = "ml_models/product_enrichment/weight_model.pkl"
    VECTORIZER_PATH = "ml_models/product_enrichment/weight_vectorizer.pkl"

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл {INPUT_CSV} не найден.")

    df = pd.read_csv(INPUT_CSV, sep=';').drop_duplicates()
    df.dropna(subset=['product_name', 'weight'], inplace=True)

    X = df['product_name']
    y = df['weight'].astype(str)  # Классификация по весу в строковом формате

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vec, y)

    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)

    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    print(f"✅ Модель веса обучена и сохранена:\n→ {MODEL_PATH}\n→ {VECTORIZER_PATH}")

default_args = {
    'start_date': datetime(2023, 1, 1),
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    dag_id='train_weight_model_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['model_training', 'weight'],
    description='Обучение модели веса продукта',
) as dag:

    train_model = PythonOperator(
        task_id='train_weight_model',
        python_callable=train_weight_model,
    )

    train_model

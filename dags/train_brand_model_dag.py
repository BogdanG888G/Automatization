from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


def train_brand_model():
    # Пути
    INPUT_CSV = "ml_models/product_enrichment/labeled_brand_products.csv"
    MODEL_PATH = "ml_models/product_enrichment/brand_model.pkl"
    VECTORIZER_PATH = "ml_models/product_enrichment/brand_vectorizer.pkl"

    # Проверка, существует ли CSV
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"❌ CSV-файл не найден: {INPUT_CSV}")

    # Загрузка размеченных данных
    df = pd.read_csv(INPUT_CSV, sep=';')
    df.dropna(subset=['product_name', 'brand'], inplace=True)

    if df.empty:
        raise ValueError("❌ Таблица для обучения бренда пуста!")

    X = df['product_name']
    y = df['brand']

    # Обучение
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vec, y)

    # Сохраняем
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)

    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    print(f"✅ Модель бренда обучена и сохранена:\n→ {MODEL_PATH}\n→ {VECTORIZER_PATH}")


# DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='train_brand_model',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['model_training', 'brand'],
) as dag:

    train_brand_task = PythonOperator(
        task_id='train_brand_model_task',
        python_callable=train_brand_model
    )

    train_brand_task

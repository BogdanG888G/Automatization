import os
import re
import logging
import pandas as pd
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from airflow import DAG
from airflow.operators.python import PythonOperator

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[«»""“”]', '', text)
    text = re.sub(r'[.,;:/\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def train_branch_model():
    INPUT_CSV = "ml_models/address_enrichment/labeled_branch_data.csv"  # путь к твоему csv
    MODEL_PATH = "ml_models/address_enrichment/branch_model.pkl"
    VECTORIZER_PATH = "ml_models/address_enrichment/branch_vectorizer.pkl"

    logging.info(f"📂 Текущая директория: {os.getcwd()}")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"❌ Файл не найден: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, sep=';').drop_duplicates()
    df.dropna(subset=['city', 'branch'], inplace=True)

    logging.info(f"📊 Кол-во уникальных городов: {df['city'].nunique()}")
    logging.info(f"📊 Кол-во уникальных филиалов: {df['branch'].nunique()}")

    df['cleaned_city'] = df['city'].apply(clean_text)
    df = df[df['cleaned_city'].str.len() > 1]

    branch_counts = df['branch'].value_counts()
    valid_branches = branch_counts[branch_counts >= 2].index
    df = df[df['branch'].isin(valid_branches)]

    if df.empty:
        raise ValueError("❌ После фильтрации данных по филиалам с минимум 2 примерами датасет пуст.")

    logging.info(f"📊 Кол-во уникальных филиалов после фильтрации: {df['branch'].nunique()}")

    X = df['cleaned_city']
    y = df['branch']

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        min_df=2,
        max_df=0.95
    )
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.4, random_state=42, stratify=y
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    logging.info("\n📈 Качество модели (филиал по городу):\n" + classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)
    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    logging.info(f"✅ Модель филиала сохранена: {MODEL_PATH}")
    logging.info(f"✅ Векторизатор сохранён: {VECTORIZER_PATH}")

# DAG airflow
default_args = {
    'start_date': datetime(2023, 1, 1),
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    dag_id='train_branch_model_dag',
    default_args=default_args,
    schedule_interval=None,
    max_active_runs=1,
    concurrency=1,
    catchup=False,
    tags=['model_training', 'address', 'branch'],
    description='Обучение модели для предсказания филиала по адресу (city)',
) as dag:

    train_branch_task = PythonOperator(
        task_id='train_branch_model',
        python_callable=train_branch_model,
    )

    train_branch_task

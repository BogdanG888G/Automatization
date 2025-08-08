from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
import os
import re
import logging
from datetime import datetime
from datetime import timedelta

from natasha import MorphVocab, AddrExtractor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

morph_vocab = MorphVocab()
addr_extractor = AddrExtractor(morph_vocab)

def normalize_address(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Немного упростил замену, убрал сложные паттерны
    replacements = {
        'г.': 'город ',
        'гор': 'город ',
        'ул': 'улица ',
        'пр-кт': 'проспект ',
        'респ': 'республика ',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r'[«»"“”]', '', text)
    text = re.sub(r'[.,;:/\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_natasha_city(text: str) -> str:
    try:
        matches = addr_extractor(text)
        for match in matches:
            for part in match.fact.parts:
                if part.type in ("город", "city"):
                    return part.value.lower()
    except Exception:
        return ""
    return ""

def train_city_model():
    INPUT_CSV = "ml_models/address_enrichment/labeled_address_data.csv"
    MODEL_PATH = "ml_models/address_enrichment/city_model_sgd.pkl"
    VECTORIZER_PATH = "ml_models/address_enrichment/city_vectorizer_hashing.pkl"

    logging.info(f"Загрузка данных из {INPUT_CSV} ...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл не найден: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, sep=';').sample(10000).dropna().drop_duplicates()
    logging.info(f"Исходный размер данных: {len(df)}")

    df['cleaned_address'] = df['address'].apply(normalize_address)
    df['natasha_city'] = df['address'].apply(extract_natasha_city)

    df = df[df['cleaned_address'].str.len() > 5]

    city_counts = df['city'].value_counts()
    df = df[df['city'].isin(city_counts[city_counts >= 3].index)]
    logging.info(f"После фильтрации по длине и частоте городов: {len(df)} строк, {df['city'].nunique()} уникальных городов")

    df['full_features'] = df['cleaned_address'] + " natasha:" + df['natasha_city']

    X = df['full_features']
    y = df['city']

    vectorizer = HashingVectorizer(
        n_features=2**16,  # 65536 признаков
        alternate_sign=False,
        ngram_range=(1, 2)
    )
    X_vec = vectorizer.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.3, random_state=42, stratify=y
    )

    model = SGDClassifier(
        loss='log',
        max_iter=1000,
        tol=1e-3,
        n_jobs=-1,
        class_weight='balanced',
        random_state=42
    )
    logging.info("Начинаем обучение модели SGDClassifier...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    logging.info(f"\nОтчёт по модели:\n{report}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)
    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    logging.info("✅ Модель и векторизатор сохранены.")

default_args = {
    'start_date': datetime(2023, 1, 1),
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='train_city_model_dag',
    default_args=default_args,
    schedule_interval=None,
    max_active_runs=1,
    concurrency=1,
    catchup=False,
    tags=['model_training', 'address', 'city'],
    description='Обучение модели для извлечения города из адреса с использованием Natasha',
) as dag:

    train_model_task = PythonOperator(
        task_id='train_city_model',
        python_callable=train_city_model,
    )

    train_model_task

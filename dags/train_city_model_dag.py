# 👇 Импорты
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
import os
import re
import logging  

from natasha import (
    MorphVocab,
    AddrExtractor
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import os
import re
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from natasha import MorphVocab, AddrExtractor

# Natasha init
morph_vocab = MorphVocab()
addr_extractor = AddrExtractor(morph_vocab)

def normalize_address(text: str) -> str:
    """Приведение адреса к единому виду"""

    if not isinstance(text, str):
        return ""
    text = text.lower()
    replace_dict = {
        r'\bг\.\b': 'город ',
        r'\bгор\b': 'город ',
        r'\bул\b': 'улица ',
        r'\bпр-кт\b': 'проспект ',
        r'\bресп\b': 'республика ',
    }
    for pattern, repl in replace_dict.items():
        text = re.sub(pattern, repl, text)
    text = re.sub(r'[«»"“”]', '', text)
    text = re.sub(r'[.,;:/\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_natasha_city(text: str) -> str:
    """Извлекает город с помощью Natasha"""

    try:
        matches = addr_extractor(text)
        for match in matches:
            for part in match.fact.parts:
                if part.type in ("город", "city"):
                    return part.value.lower()
    except Exception:
        pass
    return ""

def train_city_model():
    INPUT_CSV = "ml_models/address_enrichment/labeled_address_data.csv"
    MODEL_PATH = "ml_models/address_enrichment/city_model.pkl"
    VECTORIZER_PATH = "ml_models/address_enrichment/city_vectorizer.pkl"

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл не найден: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, sep=';').sample(10000).dropna().drop_duplicates()
    
    
    logging.info('Приводим адреса к единому виду')
    # Очистка адресов
    df['cleaned_address'] = df['address'].apply(normalize_address)
    logging.info('Извлекает город с помощью Natasha')
    
    
    # Natasha фича
    df['natasha_city'] = df['address'].apply(extract_natasha_city)
    logging.info('Извлечение закончено')

    # Фильтр по длине адреса
    df = df[df['cleaned_address'].str.len() > 5]

    # Убираем очень редкие города
    city_counts = df['city'].value_counts()
    df = df[df['city'].isin(city_counts[city_counts >= 3].index)]

    logging.info(f"Осталось {len(df)} строк, {df['city'].nunique()} уникальных городов")

    # Признаки: текст адреса + город Natasha
    df['full_features'] = df['cleaned_address'] + " natasha:" + df['natasha_city']

    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 6),
        max_features=20000
    )
    X_vec = vectorizer.fit_transform(df['full_features'])
    y = df['city']

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(
        C=3.0,
        max_iter=1500,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    logging.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)
    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    logging.info("✅ Модель и векторизатор сохранены.")


# DAG
default_args = {
    'start_date': datetime(2023, 1, 1),
    'owner': 'airflow',
    'retries': 1,
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

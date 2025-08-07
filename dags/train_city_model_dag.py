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

# 👇 Natasha init
morph_vocab = MorphVocab()
addr_extractor = AddrExtractor(morph_vocab)


def extract_city_with_natasha(text):
    if pd.isna(text) or not isinstance(text, str):
        return None

    # Попробуем регулярками
    simple_patterns = [
        r'\bг\.\s*([А-Яа-яЁё-]+)',
        r'\bгород\s*([А-Яа-яЁё-]+)',
        r'\bг\s+([А-Яа-яЁё-]+)',
        r',\s*([А-Яа-яЁё-]+)\s*,'
    ]
    for pattern in simple_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    try:
        matches = addr_extractor(text)
        for match in matches:
            fact_dict = match.fact.as_json
            # Пример: {'parts': [{'type': 'region', 'value': 'Московская'}, {'type': 'city', 'value': 'Москва'}]}
            for part in fact_dict.get("parts", []):
                if part.get("type") in ("город", "city"):
                    return part.get("value")
    except Exception as e:
        logging.warning(f"Ошибка Natasha для адреса: {text[:50]}... - {str(e)}")

    return None



def clean_address(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[«»""“”]', '', text)
    text = re.sub(r'[.,;:/\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def train_city_model():
    INPUT_CSV = "ml_models/address_enrichment/labeled_address_data.csv"
    MODEL_PATH = "ml_models/address_enrichment/city_model.pkl"
    VECTORIZER_PATH = "ml_models/address_enrichment/city_vectorizer.pkl"

    logging.info(f"📂 Рабочая директория: {os.getcwd()}")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"❌ Файл не найден: {INPUT_CSV}")

    try:
        df = pd.read_csv(INPUT_CSV, sep=';').dropna().drop_duplicates().head(10000)
        logging.info(f"🔍 Всего строк: {len(df)}")
    except Exception as e:
        raise ValueError(f"❌ Ошибка при загрузке CSV: {str(e)}")

    df.dropna(subset=['address', 'city'], inplace=True)

    logging.info("🔍 Извлекаем города с помощью Natasha...")
    df['natasha_city'] = df['address'].apply(extract_city_with_natasha)

    df['natasha_correct'] = df.apply(
        lambda x: str(x['natasha_city']).lower() == str(x['city']).lower()
        if pd.notna(x['natasha_city']) else False,
        axis=1
    )
    accuracy = df['natasha_correct'].mean()
    logging.info(f"📊 Точность Natasha: {accuracy:.2%}")

    if accuracy < 0.7:
        logging.warning("⚠️ Внимание: Точность Natasha ниже 70%. Проверьте качество данных.")

    mismatches = df[~df['natasha_correct']][['address', 'city', 'natasha_city']].head(10)
    logging.info(f"🔢 Примеры расхождений:\n{mismatches.to_string(index=False)}")

    df['final_city'] = df['natasha_city'].fillna(df['city'])
    df['cleaned_address'] = df['address'].apply(clean_address)
    df = df[df['cleaned_address'].str.len() > 5]

    city_counts = df['final_city'].value_counts()
    df = df[df['final_city'].isin(city_counts[city_counts >= 2].index)]

    if df.empty:
        raise ValueError("❌ После очистки датафрейм пуст.")

    logging.info(f"📊 Уникальных городов: {df['final_city'].nunique()}")
    logging.info(f"🏙️ Топ-10 городов:\n{df['final_city'].value_counts().head(10)}")

    X = df['cleaned_address']
    y = df['final_city']

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        min_df=2,
        max_df=0.9
    )
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.4, random_state=42, stratify=y
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    logging.info("\n📈 Качество модели:\n" + classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)
    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    logging.info(f"✅ Модель сохранена: {MODEL_PATH}")
    logging.info(f"✅ Векторизатор сохранён: {VECTORIZER_PATH}")


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

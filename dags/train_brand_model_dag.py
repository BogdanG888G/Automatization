from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
import pickle
from datetime import datetime
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+[ггмлштлк]*', ' ', text)  # удаление веса, объема
    text = re.sub(r'[^a-zа-яё\s]', ' ', text)     # только буквы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_brand_model():
    # Пути
    INPUT_CSV = "ml_models/product_enrichment/labeled_brand_products.csv"
    MODEL_PATH = "ml_models/product_enrichment/brand_model.pkl"
    VECTORIZER_PATH = "ml_models/product_enrichment/brand_vectorizer.pkl"

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"❌ CSV-файл не найден: {INPUT_CSV}")

    # Загрузка и подготовка данных
    df = pd.read_csv(INPUT_CSV, sep=';')
    df = df.drop_duplicates()
    df.dropna(subset=['product_name', 'brand'], inplace=True)
    df['product_name'] = df['product_name'].astype(str).apply(clean_text)
    df['brand'] = df['brand'].astype(str).str.title().str.strip()

    # 🔧 Удаляем редкие бренды
    brand_counts = df['brand'].value_counts()
    df = df[df['brand'].isin(brand_counts[brand_counts >= 2].index)]

    X = df['product_name']
    y = df['brand']

    # Векторизация
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=7000,
        min_df=2,
        max_df=0.9,
        stop_words=['шт', 'уп', 'г', 'гр', 'мл']
    )
    X_vec = vectorizer.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    # Обучение модели
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("📊 Отчет по метрикам качества модели:\n", report)

    # Сохранение модели и векторайзера
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

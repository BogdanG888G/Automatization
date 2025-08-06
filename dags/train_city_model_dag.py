from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def clean_address(text):
    if pd.isna(text):
        return ""
    text = text.lower()

    # Удаляем спецсимволы
    text = re.sub(r'[«»""“”]', '', text)
    text = re.sub(r'[.,;:/\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Убираем "шумовые" слова
    noise_words = [
        'ул', 'дом', 'корпус', 'кв', 'офис', 'помещ', 'строение', 'стр',
        'проезд', 'пр', 'пер', 'мкр', '№', 'здание', 'владение', 'к', 'д', 'г',
        'обл', 'р-н', 'рп', 'с', 'п', 'пгт', 'АО', 'П', '/', 'H'  # гео-сокращения
    ]
    for word in noise_words:
        text = re.sub(rf'\b{word}\b', '', text)

    return text.strip()


def train_city_model():
    # Пути
    INPUT_CSV = "ml_models/address_enrichment/labeled_address_data.csv"
    MODEL_PATH = "ml_models/address_enrichment/city_model.pkl"
    VECTORIZER_PATH = "ml_models/address_enrichment/city_vectorizer.pkl"

    print(f"📂 Рабочая директория: {os.getcwd()}")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"❌ Файл не найден: {INPUT_CSV}")

    # Загрузка
    df = pd.read_csv(INPUT_CSV, sep=';')
    print(f"🔍 Всего строк: {len(df)}")

    # Очистка
    df.dropna(subset=['address', 'city'], inplace=True)
    df['cleaned_address'] = df['address'].apply(clean_address)
    df = df[df['cleaned_address'].str.len() > 5]

    # Удаляем города, которые встречаются < 2 раз
    city_counts = df['city'].value_counts()
    df = df[df['city'].isin(city_counts[city_counts >= 2].index)]

    if df.empty:
        raise ValueError("❌ После очистки датафрейм пуст.")

    print(f"📊 Уникальных городов: {df['city'].nunique()}")
    print(f"🏙️ Топ-10 городов:\n{df['city'].value_counts().head(10)}")

    # Обучающие данные
    X = df['cleaned_address']
    y = df['city']

    # Векторизация
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20000,
        min_df=2,
        max_df=0.95
    )
    X_vec = vectorizer.fit_transform(X)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    # Модель
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    print("📈 Качество модели:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Сохранение
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)
    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    print(f"✅ Модель сохранена: {MODEL_PATH}")
    print(f"✅ Векторизатор сохранён: {VECTORIZER_PATH}")


# Аргументы DAG
default_args = {
    'start_date': datetime(2023, 1, 1),
    'owner': 'airflow',
    'retries': 1,
}

# DAG
with DAG(
    dag_id='train_city_model_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['model_training', 'address', 'city'],
    description='Обучение модели для извлечения города из адреса (только по полю address)',
) as dag:

    train_model_task = PythonOperator(
        task_id='train_city_model',
        python_callable=train_city_model,
    )

    train_model_task

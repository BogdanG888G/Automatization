from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def train_city_model():
    # Пути
    INPUT_CSV = "ml_models/address_enrichment/labeled_address_data.csv"
    MODEL_PATH = "ml_models/address_enrichment/city_model.pkl"
    VECTORIZER_PATH = "ml_models/address_enrichment/city_vectorizer.pkl"

    print(f"📂 Текущая рабочая директория: {os.getcwd()}")

    # Проверка наличия файла
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл не найден: {INPUT_CSV}")

    # Загрузка данных
    df = pd.read_csv(INPUT_CSV, sep=';')
    print(f"🔍 Прочитано строк: {len(df)}")
    print(df.head())

    # Очистка
    df.dropna(subset=['address', 'city'], inplace=True)
    if df.empty:
        raise ValueError("❌ После очистки датафрейм пустой.")

    # Признаки и цель
    X = df['address']
    y = df['city']

    # Векторизация адресов
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=8000)
    X_vec = vectorizer.fit_transform(X)

    # Обучение модели
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_vec, y)

    # Сохранение модели и векторайзера
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)
    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    print(f"✅ Модель сохранена: {MODEL_PATH}")
    print(f"✅ Векторизатор сохранён: {VECTORIZER_PATH}")

# Аргументы по умолчанию
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
    description='Обучение модели для извлечения города из адреса',
) as dag:

    train_model = PythonOperator(
        task_id='train_city_model',
        python_callable=train_city_model,
    )

    train_model

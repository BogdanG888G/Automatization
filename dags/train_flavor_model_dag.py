from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

def train_flavor_model():
    import os

    INPUT_CSV = "ml_models/product_enrichment/labeled_flavor_products.csv"
    MODEL_PATH = "ml_models/product_enrichment/flavor_model.pkl"
    VECTORIZER_PATH = "ml_models/product_enrichment/flavor_vectorizer.pkl"

    print(f"Текущая рабочая директория: {os.getcwd()}")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл {INPUT_CSV} не найден.")

    df = pd.read_csv(INPUT_CSV, sep=';')
    print(f"Размер исходных данных: {df.shape}")
    print(f"Первые 5 строк данных:\n{df.head()}")

    df.dropna(subset=['product_name', 'flavor'], inplace=True)
    print(f"Размер данных после dropna по ['product_name', 'flavor']: {df.shape}")
    if df.empty:
        raise ValueError("После удаления пропусков датафрейм пустой.")

    X = df['product_name']
    y = df['flavor']

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X_vec = vectorizer.fit_transform(X)
    print(f"Размер обучающего признакового пространства: {X_vec.shape}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vec, y)
    print(f"Модель обучена: {'estimators_' in dir(model)}")

    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)
    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    size_model = os.path.getsize(MODEL_PATH)
    size_vectorizer = os.path.getsize(VECTORIZER_PATH)
    print(f"✅ Модель сохранена: {MODEL_PATH} (размер: {size_model} байт)")
    print(f"✅ Векторизатор сохранён: {VECTORIZER_PATH} (размер: {size_vectorizer} байт)")


default_args = {
    'start_date': datetime(2023, 1, 1),
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    dag_id='train_flavor_model_dag',
    default_args=default_args,
    schedule_interval=None,  # запускаем вручную
    catchup=False,
    tags=['model_training', 'flavor'],
    description='Обучение модели по вкусам',
) as dag:

    train_model = PythonOperator(
        task_id='train_flavor_model',
        python_callable=train_flavor_model,
    )

    train_model

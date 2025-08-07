from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
import os

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def train_type_model():
    INPUT_CSV = "ml_models/product_enrichment/labeled_type_products.csv"
    MODEL_PATH = "ml_models/product_enrichment/type_model.pkl"
    VECTORIZER_PATH = "ml_models/product_enrichment/type_vectorizer.pkl"

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл {INPUT_CSV} не найден.")

    df = pd.read_csv(INPUT_CSV, sep=';')
    df.dropna(subset=['product_name', 'product_type'], inplace=True)
    if df.empty:
        raise ValueError("Датасет пуст после удаления пропусков.")

    df['product_name'] = df['product_name'].apply(preprocess_text)

    X = df['product_name']
    y = df['product_type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        stop_words='russian',
        min_df=3,
        max_df=0.9
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
        class_weight='balanced'
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("Accuracy на тесте:", accuracy_score(y_test, y_pred))
    print("Отчёт классификации:\n", classification_report(y_test, y_pred))

    with open(MODEL_PATH, 'wb') as f_model:
        pickle.dump(model, f_model)
    with open(VECTORIZER_PATH, 'wb') as f_vec:
        pickle.dump(vectorizer, f_vec)

    print(f"✅ Модель типа чипсов обучена и сохранена:\n→ {MODEL_PATH}\n→ {VECTORIZER_PATH}")

default_args = {
    'start_date': datetime(2023, 1, 1),
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    dag_id='train_type_model_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['model_training', 'type'],
    description='Оптимизированное обучение модели типа чипсов',
) as dag:

    train_model = PythonOperator(
        task_id='train_type_model',
        python_callable=train_type_model,
    )

    train_model

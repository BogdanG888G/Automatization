from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import logging
import pickle
import re
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq
from pyarrow import Table
import csv
import numpy as np
from sklearn.utils import gen_batches
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Конфигурация
CONN_STRING = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)

DATA_DIR = "/opt/airflow/data"
CHUNK_SIZE = 50_000  # Уменьшили для стабильности
ML_MODELS_DIR = "/opt/airflow/ml_models"
TEMP_PARQUET_DIR = "/opt/airflow/temp_parquet"

# Создаем временную директорию для parquet файлов
os.makedirs(TEMP_PARQUET_DIR, exist_ok=True)

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def get_engine():
    """Создает engine для подключения к БД"""
    return create_engine(CONN_STRING)

def load_all_models():
    """Загружает ВСЕ 7 моделей"""
    ml_models = {}
    
    try:
        # Загружаем модели для продуктов
        product_models_dir = os.path.join(ML_MODELS_DIR, "product_enrichment")
        if os.path.exists(product_models_dir):
            ml_features = ['flavor', 'brand', 'weight', 'type']
            for model_name in ml_features:
                try:
                    model_path = os.path.join(product_models_dir, f"{model_name}_model.pkl")
                    vectorizer_path = os.path.join(product_models_dir, f"{model_name}_vectorizer.pkl")
                    
                    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        with open(vectorizer_path, 'rb') as f:
                            vectorizer = pickle.load(f)
                        
                        ml_models[f"product_{model_name}"] = (model, vectorizer)
                        logging.info(f"Загружена продуктовая модель: {model_name}")
                    else:
                        logging.warning(f"Файлы модели {model_name} не найдены")
                except Exception as e:
                    logging.error(f"Ошибка загрузки модели {model_name}: {e}")

        # Загружаем модели для адресов
        address_models_dir = os.path.join(ML_MODELS_DIR, "address_enrichment")
        if os.path.exists(address_models_dir):
            address_models = ['city', 'region', 'branch']
            for model_name in address_models:
                try:
                    model_path = os.path.join(address_models_dir, f"{model_name}_model.pkl")
                    vectorizer_path = os.path.join(address_models_dir, f"{model_name}_vectorizer.pkl")
                    
                    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        with open(vectorizer_path, 'rb') as f:
                            vectorizer = pickle.load(f)
                        
                        ml_models[f"address_{model_name}"] = (model, vectorizer)
                        logging.info(f"Загружена адресная модель: {model_name}")
                    else:
                        logging.warning(f"Файлы модели {model_name} не найдены")
                except Exception as e:
                    logging.error(f"Ошибка загрузки модели {model_name}: {e}")
        
        logging.info(f"Всего загружено моделей: {len(ml_models)}")
        
    except Exception as e:
        logging.error(f"Ошибка загрузки моделей: {e}")
    
    return ml_models

# Компилируем regex заранее для производительности
WEIGHT_PATTERN = re.compile(r'(\d+(?:[.,]\d+)?)[ ]?(г|гр|грамм|кг|кгм)', re.IGNORECASE)

def extract_weight(text):
    """Извлекает вес из текста и конвертирует в граммы"""
    if not isinstance(text, str):
        return None
    
    match = WEIGHT_PATTERN.search(text)
    if match:
        number = match.group(1).replace(',', '.')
        unit = match.group(2).lower()
        try:
            weight_val = float(number)
            # Конвертируем в граммы если указаны кг
            if any(kg_unit in unit for kg_unit in ['кг', 'кгм']):
                weight_val *= 1000
            return int(weight_val)
        except (ValueError, TypeError):
            return None
    return None

def extract_packaging_type(product_name):
    """Определяет тип упаковки продукта"""
    if not isinstance(product_name, str):
        return "Пакет"
    
    text = product_name.lower()

    # Ключевые слова для туб
    tube_keywords = ["туба", "тубус", "tube", "can"]
    tube_brands = ["pringles", "stax", "lays stax", "big bon chips", "just brutal"]

    # Проверяем ключевые слова
    if any(word in text for word in tube_keywords):
        return "Туба"

    # Проверяем бренды
    if any(brand in text for brand in tube_brands):
        return "Туба"
    
    return "Пакет"

def ensure_magnit_schema(engine):
    """Создает схему magnit если она не существует"""
    try:
        with engine.connect() as conn:
            # Проверяем существование схемы magnit
            check_schema_sql = """
            SELECT COUNT(*) 
            FROM sys.schemas 
            WHERE name = 'magnit'
            """
            result = conn.execute(text(check_schema_sql)).scalar()
            
            if result == 0:
                # Создаем схему magnit
                create_schema_sql = "CREATE SCHEMA magnit"
                conn.execute(text(create_schema_sql))
                logging.info("Схема magnit создана")
            else:
                logging.info("Схема magnit уже существует")
                
            return True
            
    except Exception as e:
        logging.error(f"Ошибка создания схемы magnit: {e}")
        return False

def create_table_if_not_exists(engine, table_name, sample_chunk):
    """Создает таблицу в схеме magnit если она не существует"""
    
    # Сначала убеждаемся, что схема magnit существует
    if not ensure_magnit_schema(engine):
        return False
    
    # Все столбцы создаем как NVARCHAR(255)
    columns_sql = []
    for col_name in sample_chunk.columns:
        columns_sql.append(f"[{col_name}] NVARCHAR(255)")
    
    create_table_sql = f"""
    IF NOT EXISTS (SELECT * FROM sys.tables t 
                   JOIN sys.schemas s ON t.schema_id = s.schema_id 
                   WHERE t.name = '{table_name}' AND s.name = 'magnit')
    BEGIN
        CREATE TABLE [magnit].[{table_name}] (
            {', '.join(columns_sql)}
        )
    END
    """
    
    try:
        with engine.connect() as conn:
            # Создаем таблицу
            conn.execute(text(create_table_sql))
            
        logging.info(f"Таблица magnit.{table_name} создана или уже существует")
        return True
        
    except Exception as e:
        logging.error(f"Ошибка создания таблицы magnit.{table_name}: {e}")
        return False

def standardize_column_names_complete(df):
    """Полная стандартизация названий колонок"""
    column_mapping = {}
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Обрабатываем BOM символ и кавычки
        clean_col = col_lower.replace('\ufeff', '').replace('"', '')
        
        if 'наименование' in clean_col and any(keyword in clean_col for keyword in ['поставщика', 'поставщик']):
            column_mapping[col] = 'supplier_name'
        elif 'наименование' in clean_col:
            column_mapping[col] = 'product_name'
        elif 'адрес' in clean_col:
            column_mapping[col] = 'address'
        elif 'город' in clean_col:
            column_mapping[col] = 'city'
        elif 'регион' in clean_col:
            column_mapping[col] = 'region'
        elif any(keyword in clean_col for keyword in ['филиал', 'отделение']):
            column_mapping[col] = 'branch'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logging.info(f"Переименованы колонки: {column_mapping}")
    
    return df

def safe_to_sql_fixed(df, table_name, engine, schema='magnit', if_exists='append'):
    """Исправленная безопасная вставка в SQL"""
    try:
        # Убедимся, что все значения строковые и не слишком длинные
        for col in df.columns:
            df[col] = df[col].astype(str).str.slice(0, 250)  # Ограничиваем длину
        
        # Разбиваем на очень маленькие части для избежания ошибки параметров
        batch_size = 1000  # Уменьшили размер батча
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            chunk = df.iloc[i:i + batch_size]
            chunk.to_sql(
                name=table_name,
                con=engine,
                schema=schema,
                if_exists=if_exists if i == 0 else 'append',
                index=False,
                method=None  # Убираем method='multi' чтобы избежать ошибки параметров
            )
            logging.info(f"Успешно вставлен батч {i//batch_size + 1}/{total_batches}")
        
        return True
    except Exception as e:
        logging.error(f"Ошибка вставки в SQL: {e}")
        return False

def predict_in_batches_fast(model, vectorizer, texts, batch_size=5000):
    """Быстрое предсказание батчами"""
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            features = vectorizer.transform(batch_texts)
            batch_predictions = model.predict(features)
            predictions.extend(batch_predictions)
        except Exception as e:
            logging.error(f"Ошибка в батче {i//batch_size + 1}: {e}")
            predictions.extend([""] * len(batch_texts))
    
    return predictions

def enrich_data_complete(df, ml_models):
    """Полное обогащение данных со ВСЕМИ 7 моделями"""
    try:
        logging.info(f"Начинаем ПОЛНОЕ обогащение данных. Моделей: {len(ml_models)}")
        
        # Стандартизация колонок
        df = standardize_column_names_complete(df)
        logging.info(f"Колонки после стандартизации: {list(df.columns)}")
        
        # Базовые признаки
        if 'product_name' in df.columns:
            df['weight_extracted_regex'] = df['product_name'].apply(extract_weight)
            df['package_type'] = df['product_name'].apply(extract_packaging_type)
            logging.info("Базовые признаки извлечены")
            
            # ПРОДУКТОВЫЕ МОДЕЛИ (4 модели)
            product_texts = df['product_name'].fillna('').astype(str).tolist()
            
            product_mappings = [
                ('product_flavor', 'flavor_predicted'),
                ('product_brand', 'brand_predicted'), 
                ('product_weight', 'weight_predicted'),
                ('product_type', 'product_type_predicted')
            ]
            
            for model_key, output_col in product_mappings:
                if model_key in ml_models:
                    logging.info(f"Применяем продуктовую модель: {model_key}")
                    model, vectorizer = ml_models[model_key]
                    try:
                        df[output_col] = predict_in_batches_fast(model, vectorizer, product_texts, batch_size=2000)
                        logging.info(f"Успешно применена модель {model_key}")
                    except Exception as e:
                        logging.error(f"Ошибка применения модели {model_key}: {e}")
                        df[output_col] = ""
                else:
                    logging.warning(f"Модель {model_key} не найдена")
                    df[output_col] = ""
            
            # АДРЕСНЫЕ МОДЕЛИ (3 модели)
            if 'address' in df.columns:
                logging.info("Применяем адресные модели")
                address_texts = df['address'].fillna('').astype(str).tolist()
                
                # Модель города на основе адреса
                if 'address_city' in ml_models:
                    logging.info("Применяем модель города")
                    city_model, city_vectorizer = ml_models['address_city']
                    try:
                        df['city_predicted'] = predict_in_batches_fast(city_model, city_vectorizer, address_texts, batch_size=2000)
                    except Exception as e:
                        logging.error(f"Ошибка предсказания города: {e}")
                        df['city_predicted'] = ""
                else:
                    df['city_predicted'] = ""
                
                # Модель региона и филиала на основе предсказанного города
                if 'city_predicted' in df.columns:
                    city_predicted_texts = df['city_predicted'].fillna('').astype(str).tolist()
                    
                    if 'address_region' in ml_models:
                        logging.info("Применяем модель региона")
                        region_model, region_vectorizer = ml_models['address_region']
                        try:
                            df['region_predicted'] = predict_in_batches_fast(region_model, region_vectorizer, city_predicted_texts, batch_size=5000)
                        except Exception as e:
                            logging.error(f"Ошибка предсказания региона: {e}")
                            df['region_predicted'] = ""
                    else:
                        df['region_predicted'] = ""
                    
                    if 'address_branch' in ml_models:
                        logging.info("Применяем модель филиала")
                        branch_model, branch_vectorizer = ml_models['address_branch']
                        try:
                            df['branch_predicted'] = predict_in_batches_fast(branch_model, branch_vectorizer, city_predicted_texts, batch_size=5000)
                        except Exception as e:
                            logging.error(f"Ошибка предсказания филиала: {e}")
                            df['branch_predicted'] = ""
                    else:
                        df['branch_predicted'] = ""
                else:
                    df['region_predicted'] = ""
                    df['branch_predicted'] = ""
            else:
                logging.warning("Колонка address не найдена для адресных моделей")
                df['city_predicted'] = ""
                df['region_predicted'] = ""
                df['branch_predicted'] = ""
                
        else:
            logging.warning("Не найдена колонка product_name")
            # Создаем все колонки с пустыми значениями
            ml_columns = [
                'weight_extracted_regex', 'package_type',
                'flavor_predicted', 'brand_predicted', 'weight_predicted', 'product_type_predicted',
                'city_predicted', 'region_predicted', 'branch_predicted'
            ]
            for col in ml_columns:
                df[col] = ""

        # Логируем результат
        added_columns = [col for col in df.columns if 'predicted' in col or 'extracted' in col]
        logging.info(f"Обогащение завершено. Добавлено колонок: {len(added_columns)}")
        logging.info(f"Добавленные колонки: {added_columns}")
        
        return df

    except Exception as e:
        logging.error(f"Ошибка в полном обогащении: {e}")
        return df

def process_csv_with_all_models(file_path, engine, table_name):
    """Обработка CSV со ВСЕМИ моделями"""
    try:
        logging.info(f"Начинаем обработку со ВСЕМИ моделями: {file_path}")
        
        # Загружаем ВСЕ модели
        ml_models = load_all_models()
        
        chunk_count = 0
        total_rows = 0
        first_chunk = True
        
        for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE, delimiter=';', encoding='utf-8')):
            # Базовая очистка
            chunk = chunk.dropna(how='all')
            chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]
            
            if not chunk.empty:
                logging.info(f"Обрабатываем чанк {chunk_num}: {len(chunk)} строк")
                
                # ПОЛНОЕ обогащение со всеми моделями
                start_time = datetime.now()
                enriched_chunk = enrich_data_complete(chunk, ml_models)
                processing_time = (datetime.now() - start_time).total_seconds()
                
                logging.info(f"Обогащение чанка {chunk_num} заняло {processing_time:.2f} секунд")
                
                # Конвертируем в строки
                for col in enriched_chunk.columns:
                    enriched_chunk[col] = enriched_chunk[col].astype(str)
                
                # Создаем таблицу при первом чанке
                if first_chunk:
                    if create_table_if_not_exists(engine, table_name, enriched_chunk):
                        logging.info(f"Таблица magnit.{table_name} готова")
                    first_chunk = False
                
                # Безопасная вставка с исправленной функцией
                if safe_to_sql_fixed(enriched_chunk, table_name, engine, if_exists='replace' if chunk_num == 0 else 'append'):
                    chunk_count += 1
                    total_rows += len(enriched_chunk)
                    logging.info(f"УСПЕШНО загружен чанк {chunk_num} в magnit.{table_name}")
                else:
                    logging.error(f"ОШИБКА загрузки чанка {chunk_num}")
                    # Пробуем альтернативный метод
                    try:
                        enriched_chunk.to_sql(
                            name=table_name,
                            con=engine,
                            schema='magnit',
                            if_exists='append',
                            index=False,
                            chunksize=500
                        )
                        logging.info(f"Загрузка через альтернативный метод успешна")
                        chunk_count += 1
                        total_rows += len(enriched_chunk)
                    except Exception as e:
                        logging.error(f"Альтернативный метод тоже не сработал: {e}")
        
        return chunk_count, total_rows
        
    except Exception as e:
        logging.error(f"Ошибка в обработке CSV: {e}")
        return 0, 0

@task
def scan_files():
    """Сканирует CSV файлы с префиксом aushan_ в директории"""
    csv_files = []
    try:
        for file in os.listdir(DATA_DIR):
            if file.startswith('aushan_') and file.endswith('.csv'):
                file_path = os.path.join(DATA_DIR, file)
                if os.path.isfile(file_path):
                    csv_files.append(file_path)
        
        logging.info(f"Найдено {len(csv_files)} CSV файлов: {[os.path.basename(f) for f in csv_files]}")
        return csv_files
        
    except Exception as e:
        logging.error(f"Ошибка сканирования файлов: {e}")
        return []

@task
def process_file_complete(file_path):
    """Обработка файла со ВСЕМИ моделями"""
    if not os.path.exists(file_path):
        logging.error(f"Файл не найден: {file_path}")
        return
        
    engine = get_engine()
    original_filename = os.path.basename(file_path).replace('.csv', '')
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', original_filename.lower())
    
    try:
        logging.info(f"=== НАЧАЛО ПОЛНОЙ ОБРАБОТКИ {file_path} ===")
        
        # Используем обработку со всеми моделями
        chunk_count, total_rows = process_csv_with_all_models(file_path, engine, table_name)
        
        if total_rows > 0:
            logging.info(f"=== УСПЕХ: Обработан {file_path} ===")
            logging.info(f"Чанков: {chunk_count}, Строк: {total_rows}")
            logging.info(f"Таблица: magnit.{table_name}")
            
            # Проверяем что данные действительно в базе
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM magnit.{table_name}"))
                    count_in_db = result.scalar()
                    logging.info(f"ПРОВЕРКА: В таблице magnit.{table_name} сейчас {count_in_db} строк")
                    
                    # Проверяем несколько записей
                    sample_result = conn.execute(text(f"SELECT TOP 3 product_name, brand_predicted, city_predicted FROM magnit.{table_name}"))
                    sample_rows = sample_result.fetchall()
                    logging.info("ПРИМЕРЫ ДАННЫХ:")
                    for row in sample_rows:
                        logging.info(f"  Продукт: {row[0]}, Бренд: {row[1]}, Город: {row[2]}")
                        
            except Exception as e:
                logging.warning(f"Не удалось проверить данные в БД: {e}")
        else:
            logging.error(f"=== ПРОВАЛ: Файл {file_path} не обработан ===")
        
    except Exception as e:
        logging.error(f"КРИТИЧЕСКАЯ ОШИБКА обработки {file_path}: {str(e)}")
        raise
    finally:
        engine.dispose()

with DAG(
    dag_id="aushan_sales_pipeline_complete",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_tasks=1,
    tags=["aushan", "complete", "all_models"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    files = scan_files()
    process_tasks = process_file_complete.expand(file_path=files)

    start >> files >> process_tasks >> end
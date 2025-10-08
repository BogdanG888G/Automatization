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
import shutil
warnings.filterwarnings('ignore', category=UserWarning)

# Конфигурация
CONN_STRING = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)
DATA_DIR = "/opt/airflow/data"
ARCHIVE_DIR = "/opt/airflow/archive"
CHUNK_SIZE = 50_000
ML_MODELS_DIR = "/opt/airflow/ml_models"
TEMP_PARQUET_DIR = "/opt/airflow/temp_parquet"

# Создаем временные директории
os.makedirs(TEMP_PARQUET_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# ФАЙЛОВЫЙ КЭШ ДЛЯ МОДЕЛЕЙ (работает в Airflow)
MODELS_CACHE_DIR = "/opt/airflow/models_cache"
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def get_engine():
    """Создает engine для подключения к БД"""
    return create_engine(CONN_STRING)

def load_all_models_cached():
    """Загружает ВСЕ 7 моделей с ФАЙЛОВЫМ кэшированием"""
    cache_file = os.path.join(MODELS_CACHE_DIR, "models_cache.pkl")
    cache_meta_file = os.path.join(MODELS_CACHE_DIR, "models_cache_meta.pkl")
    
    # Проверяем актуальность кэша (1 час)
    if os.path.exists(cache_file) and os.path.exists(cache_meta_file):
        try:
            with open(cache_meta_file, 'rb') as f:
                cache_meta = pickle.load(f)
            
            cache_time = cache_meta.get('timestamp')
            if cache_time and (datetime.now() - cache_time).total_seconds() < 3600:
                logging.info("Используем кэшированные модели из файла")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logging.warning(f"Ошибка чтения кэша: {e}. Перезагружаем модели.")
    
    # Загружаем модели заново
    ml_models = {}
    
    try:
        logging.info("Начинаем загрузку моделей...")
        start_time = datetime.now()
        
        # Загружаем модели для продуктов
        product_models_dir = os.path.join(ML_MODELS_DIR, "product_enrichment")
        if os.path.exists(product_models_dir):
            ml_features = ['flavor', 'brand', 'weight', 'type']
            for model_name in ml_features:
                try:
                    model_path = os.path.join(product_models_dir, f"{model_name}_model.pkl")
                    vectorizer_path = os.path.join(product_models_dir, f"{model_name}_vectorizer.pkl")
                    
                    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                        logging.info(f"Загружаем модель: {model_name}")
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        with open(vectorizer_path, 'rb') as f:
                            vectorizer = pickle.load(f)
                        
                        ml_models[f"product_{model_name}"] = (model, vectorizer)
                        logging.info(f"✅ Загружена продуктовая модель: {model_name}")
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
                        logging.info(f"Загружаем модель: {model_name}")
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        with open(vectorizer_path, 'rb') as f:
                            vectorizer = pickle.load(f)
                        
                        ml_models[f"address_{model_name}"] = (model, vectorizer)
                        logging.info(f"✅ Загружена адресная модель: {model_name}")
                    else:
                        logging.warning(f"Файлы модели {model_name} не найдены")
                except Exception as e:
                    logging.error(f"Ошибка загрузки модели {model_name}: {e}")
        
        # Сохраняем в файловый кэш
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(ml_models, f)
            with open(cache_meta_file, 'wb') as f:
                pickle.dump({'timestamp': datetime.now()}, f)
            logging.info("Модели сохранены в файловый кэш")
        except Exception as e:
            logging.warning(f"Не удалось сохранить кэш: {e}")
        
        load_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Всего загружено моделей: {len(ml_models)} за {load_time:.2f} секунд")
        
    except Exception as e:
        logging.error(f"Ошибка загрузки моделей: {e}")
    
    return ml_models

def archive_file(file_path):
    """Просто перемещает файл в папку archive"""
    try:
        if not os.path.exists(file_path):
            logging.warning(f"Файл для архивации не найден: {file_path}")
            return False
        
        filename = os.path.basename(file_path)
        
        # Просто перемещаем файл в корень archive
        archive_path = os.path.join(ARCHIVE_DIR, filename)
        
        # Если файл с таким именем уже существует, добавляем timestamp
        if os.path.exists(archive_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_part = os.path.splitext(filename)[0]
            extension = os.path.splitext(filename)[1]
            new_filename = f"{name_part}_{timestamp}{extension}"
            archive_path = os.path.join(ARCHIVE_DIR, new_filename)
        
        # Перемещаем файл
        shutil.move(file_path, archive_path)
        logging.info(f"Файл перемещен в архив: {file_path} -> {archive_path}")
        
        # Проверяем, что файл действительно переместился
        if os.path.exists(archive_path) and not os.path.exists(file_path):
            logging.info(f"✅ Подтверждено: файл успешно архивирован")
            return True
        else:
            logging.error(f"❌ Ошибка архивации: файл не переместился корректно")
            return False
        
    except Exception as e:
        logging.error(f"Ошибка архивации файла {file_path}: {e}")
        return False

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

def ensure_x5_schema(engine):
    """Создает схему x5 если она не существует"""
    try:
        with engine.connect() as conn:
            # Проверяем существование схемы x5
            check_schema_sql = """
            SELECT COUNT(*) 
            FROM sys.schemas 
            WHERE name = 'x5'
            """
            result = conn.execute(text(check_schema_sql)).scalar()
            
            if result == 0:
                # Создаем схему x5
                create_schema_sql = "CREATE SCHEMA x5"
                conn.execute(text(create_schema_sql))
                logging.info("Схема x5 создана")
            else:
                logging.info("Схема x5 уже существует")
                
            return True
            
    except Exception as e:
        logging.error(f"Ошибка создания схемы x5: {e}")
        return False

def create_table_if_not_exists(engine, table_name, sample_chunk):
    """Создает таблицу в схеме x5 если она не существует"""
    
    # Сначала убеждаемся, что схема x5 существует
    if not ensure_x5_schema(engine):
        return False
    
    # Все столбцы создаем как NVARCHAR(255)
    columns_sql = []
    for col_name in sample_chunk.columns:
        columns_sql.append(f"[{col_name}] NVARCHAR(255)")
    
    create_table_sql = f"""
    IF NOT EXISTS (SELECT * FROM sys.tables t 
                   JOIN sys.schemas s ON t.schema_id = s.schema_id 
                   WHERE t.name = '{table_name}' AND s.name = 'x5')
    BEGIN
        CREATE TABLE [x5].[{table_name}] (
            {', '.join(columns_sql)}
        )
    END
    """
    
    try:
        with engine.connect() as conn:
            # Создаем таблицу
            conn.execute(text(create_table_sql))
            
        logging.info(f"Таблица x5.{table_name} создана или уже существует")
        return True
        
    except Exception as e:
        logging.error(f"Ошибка создания таблицы x5.{table_name}: {e}")
        return False

def standardize_column_names_complete(df):
    """Полная стандартизация названий колонок с обработкой дубликатов"""
    column_mapping = {}
    used_names = set()
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Обрабатываем BOM символ и кавычки
        clean_col = col_lower.replace('\ufeff', '').replace('"', '')
        
        new_name = None
        
        if any(keyword in clean_col for keyword in ['поставщика', 'поставщик']):
            new_name = 'supplier_name'
        elif 'уни наименование' in clean_col or 'товар' in clean_col or 'наименование' in clean_col or 'материал2' in clean_col:
            new_name = 'product_name'
        elif 'код товара' in clean_col or 'артикул' in clean_col or 'код' in clean_col:
            new_name = 'product_code'  # Отдельная колонка для кода товара
        elif 'адрес' in clean_col:
            new_name = 'address'
        elif 'город' in clean_col:
            new_name = 'city'
        elif 'регион' in clean_col:
            new_name = 'region'
        elif any(keyword in clean_col for keyword in ['филиал', 'отделение']):
            new_name = 'branch'
        elif 'product' in clean_col:
            new_name = 'product_name'
        
        # Обрабатываем дубликаты
        if new_name:
            if new_name in used_names:
                # Если имя уже используется, добавляем суффикс
                counter = 1
                while f"{new_name}_{counter}" in used_names:
                    counter += 1
                final_name = f"{new_name}_{counter}"
            else:
                final_name = new_name
            
            column_mapping[col] = final_name
            used_names.add(final_name)
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logging.info(f"Переименованы колонки: {column_mapping}")
    
    return df

def safe_to_sql_fixed(df, table_name, engine, schema='x5', if_exists='append'):
    """Исправленная безопасная вставка в SQL"""
    try:
        # Убедимся, что все значения строковые и не слишком длинные
        for col in df.columns:
            # Проверяем, что колонка существует и преобразуем в строку
            if col in df.columns:
                df[col] = df[col].astype(str).str.slice(0, 250)  # Ограничиваем длину
        
        # Разбиваем на очень маленькие части для избежания ошибки параметров
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            chunk = df.iloc[i:i + batch_size]
            chunk.to_sql(
                name=table_name,
                con=engine,
                schema=schema,
                if_exists=if_exists if i == 0 else 'append',
                index=False,
                method=None
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
        
        return df

    except Exception as e:
        logging.error(f"Ошибка в полном обогащении: {e}")
        return df

def process_csv_with_all_models(file_path, engine, table_name):
    """Обработка CSV со ВСЕМИ моделями"""
    try:
        logging.info(f"Начинаем обработку со ВСЕМИ моделями: {file_path}")
        
        # Загружаем ВСЕ модели С ФАЙЛОВЫМ КЭШИРОВАНИЕМ
        ml_models = load_all_models_cached()
        
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
                        logging.info(f"Таблица x5.{table_name} готова")
                    first_chunk = False
                
                # Безопасная вставка с исправленной функцией
                if safe_to_sql_fixed(enriched_chunk, table_name, engine, if_exists='replace' if chunk_num == 0 else 'append'):
                    chunk_count += 1
                    total_rows += len(enriched_chunk)
                    logging.info(f"УСПЕШНО загружен чанк {chunk_num} в x5.{table_name}")
                else:
                    logging.error(f"ОШИБКА загрузки чанка {chunk_num}")
                    # Пробуем альтернативный метод
                    try:
                        enriched_chunk.to_sql(
                            name=table_name,
                            con=engine,
                            schema='x5',
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
    """Сканирует CSV файлы с префиксом x5_ в директории"""
    csv_files = []
    try:
        logging.info(f"=== СКАНИРУЕМ ПАПКУ {DATA_DIR} ===")
        
        if not os.path.exists(DATA_DIR):
            logging.error(f"Папка {DATA_DIR} не существует!")
            return []
        
        all_files = os.listdir(DATA_DIR)
        logging.info(f"Все файлы в папке: {all_files}")
        
        for file in all_files:
            file_path = os.path.join(DATA_DIR, file)
            if file.startswith('x5_') and file.endswith('.csv') and os.path.isfile(file_path):
                csv_files.append(file_path)
                logging.info(f"✅ Найден файл: {file}")
        
        logging.info(f"Итого найдено файлов: {len(csv_files)}")
        return csv_files
        
    except Exception as e:
        logging.error(f"Ошибка сканирования файлов: {e}")
        return []

@task
def process_file_complete(file_path):
    """Обработка файла со ВСЕМИ моделями"""
    logging.info(f"=== ОБРАБОТКА ФАЙЛА: {file_path} ===")
    
    if not os.path.exists(file_path):
        logging.error(f"Файл не найден: {file_path}")
        return
        
    engine = get_engine()
    original_filename = os.path.basename(file_path).replace('.csv', '')
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', original_filename.lower())
    
    file_processed_successfully = False
    
    try:
        logging.info(f"=== НАЧАЛО ОБРАБОТКИ {file_path} ===")
        
        # Используем обработку со всеми моделями
        chunk_count, total_rows = process_csv_with_all_models(file_path, engine, table_name)
        
        if total_rows > 0:
            logging.info(f"=== УСПЕХ: Обработан {file_path} ===")
            logging.info(f"Чанков: {chunk_count}, Строк: {total_rows}")
            logging.info(f"Таблица: x5.{table_name}")
            
            # Проверяем что данные действительно в базе
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM x5.{table_name}"))
                    count_in_db = result.scalar()
                    logging.info(f"ПРОВЕРКА: В таблице x5.{table_name} сейчас {count_in_db} строк")
                    
            except Exception as e:
                logging.warning(f"Не удалось проверить данные в БД: {e}")
            
            # АРХИВАЦИЯ только при успешной обработке
            if archive_file(file_path):
                logging.info(f"Файл перемещен в архив: {file_path}")
                file_processed_successfully = True
                
        else:
            logging.error(f"=== ПРОВАЛ: Файл {file_path} не обработан ===")
            # Архивируем даже если не обработали, но с пометкой ошибки
            if archive_file(file_path):
                logging.info(f"Файл перемещен в архив после ошибки обработки: {file_path}")
                file_processed_successfully = True
        
    except Exception as e:
        logging.error(f"Ошибка обработки {file_path}: {str(e)}")
        # Пытаемся архивировать файл даже при ошибке
        if archive_file(file_path):
            logging.info(f"Файл перемещен в архив после исключения: {file_path}")
            file_processed_successfully = True
        raise
        
    finally:
        engine.dispose()
        # Убираем дублирующий вызов archive_file из finally
        # Файл уже архивирован в основном блоке или блоке исключения

with DAG(
    dag_id="x5_sales_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_tasks=1,
    tags=["x5"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    files = scan_files()
    process_tasks = process_file_complete.expand(file_path=files)

    start >> files >> process_tasks >> end
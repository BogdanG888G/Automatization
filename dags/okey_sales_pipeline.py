from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import os
import shutil
import logging
from sqlalchemy import create_engine, exc
import pandas as pd
from typing import List
from airflow.utils.task_group import TaskGroup
import chardet
import pyxlsb  # Для поддержки xlsb

# Коннекты для тестовой и стейдж-баз
DEFAULT_CONN_TEST = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)
DEFAULT_CONN_STAGE = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)

class OkeyConfig:
    MAX_CONCURRENT_TASKS = 4
    MAX_FILES_PER_RUN = 8
    TASK_TIMEOUT = timedelta(minutes=60)
    MAX_FILE_SIZE = 3 * 1024 * 1024 * 1024  # 3GB

    DATA_DIR = "/opt/airflow/data"
    ARCHIVE_DIR = "/opt/airflow/archive"
    ALLOWED_EXT = {'.csv', '.xlsx', '.xls', '.xlsb'}
    FILE_PREFIX = "okey_"
    STAGE_SCHEMA = "okey"

    @staticmethod
    def get_processing_pool():
        try:
            pools = Variable.get("airflow_pools", default_var="default_pool")
            return "okey_file_pool" if "okey_file_pool" in pools.split(",") else "default_pool"
        except:
            return "default_pool"

    PROCESSING_POOL = get_processing_pool()
    POOL_SLOTS = 6

    CONN_SETTINGS = {
        'pool_size': 2,
        'max_overflow': 3,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'connect_args': {
            'timeout': 900,
            'application_name': 'airflow_okey_loader'
        }
    }

_engine_cache = {}
_last_cache_update = datetime.min

def get_engine(db_type: str):
    global _last_cache_update

    if (datetime.now() - _last_cache_update) > timedelta(hours=6):
        _engine_cache.clear()
        _last_cache_update = datetime.now()

    if db_type not in _engine_cache:
        try:
            if db_type.lower() == "test":
                conn_str = DEFAULT_CONN_TEST
            elif db_type.lower() == "stage":
                conn_str = DEFAULT_CONN_STAGE
            else:
                raise ValueError(f"Unknown DB type: {db_type}")

            _engine_cache[db_type] = create_engine(conn_str, **OkeyConfig.CONN_SETTINGS)
        except Exception as e:
            logging.error(f"Error creating engine for {db_type}: {str(e)}")
            raise

    return _engine_cache[db_type]

def detect_file_encoding(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'

@task(
    task_id="scan_okey_files",
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=OkeyConfig.TASK_TIMEOUT,
    pool=OkeyConfig.PROCESSING_POOL,
    pool_slots=1
)
def scan_okey_files() -> List[str]:
    valid_files = []
    total_size = 0

    logging.info(f"Scanning directory: {OkeyConfig.DATA_DIR}")
    if not os.path.exists(OkeyConfig.DATA_DIR):
        raise FileNotFoundError(f"Directory not found: {OkeyConfig.DATA_DIR}")

    for root, _, files in os.walk(OkeyConfig.DATA_DIR):
        for f in files:
            if (f.lower().startswith(OkeyConfig.FILE_PREFIX) and
                os.path.splitext(f)[1].lower() in OkeyConfig.ALLOWED_EXT):
                
                file_path = os.path.join(root, f)
                file_size = os.path.getsize(file_path)

                if file_size == 0:
                    logging.warning(f"Skipping empty file: {file_path}")
                    continue

                if total_size + file_size > OkeyConfig.MAX_FILE_SIZE:
                    logging.warning(f"File size limit exceeded. Skipping {file_path}")
                    continue

                valid_files.append(file_path)
                total_size += file_size

                if len(valid_files) >= OkeyConfig.MAX_FILES_PER_RUN:
                    break

        if len(valid_files) >= OkeyConfig.MAX_FILES_PER_RUN:
            break

    logging.info(f"Found {len(valid_files)} Okey files (total size: {total_size / 1024 / 1024:.2f} MB)")
    return valid_files[:OkeyConfig.MAX_FILES_PER_RUN]

@task(
    task_id="process_okey_file",
    retries=3,
    retry_delay=timedelta(minutes=5),
    execution_timeout=OkeyConfig.TASK_TIMEOUT,
    pool=OkeyConfig.PROCESSING_POOL,
    pool_slots=1
)
def process_okey_file(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            logging.warning(f"Skipping empty file: {file_path}")
            archive_file(file_path, empty=True)
            return

        # Здесь можно использовать функцию чтения из пятерочки или написать свою,
        # но для примера - просто попробуем загрузить первые 10000 строк
        df = pd.read_excel(file_path, engine='openpyxl', nrows=None)  # или кастомный reader
        if df is None or df.empty:
            logging.error(f"Failed to read data from file: {file_path}")
            archive_file(file_path, error=True)
            return

        engine_test = get_engine('stage')
        try:
            from okey.create_table_and_upload import create_okey_table_and_upload
            table_name = create_okey_table_and_upload(file_path, engine=engine_test)
            logging.info(f"Okey data loaded to okey.{table_name}")
        except exc.SQLAlchemyError as e:
            engine_test.dispose()
            logging.error(f"Raw load error: {str(e)}")
            archive_file(file_path, error=True)
            raise RuntimeError(f"Raw load error: {str(e)}")
        
        finally:
            engine_test.dispose()

        archive_file(file_path)

    except Exception as e:
        logging.error(f"Okey file processing error: {file_path} - {str(e)}", exc_info=True)
        archive_file(file_path, error=True)
        raise

def archive_file(file_path: str):
    os.makedirs(OkeyConfig.ARCHIVE_DIR, exist_ok=True)
    file_name = os.path.basename(file_path)  # сохраняем оригинальное имя с расширением
    archive_path = os.path.join(OkeyConfig.ARCHIVE_DIR, file_name)

    try:
        shutil.move(file_path, archive_path)
        logging.info(f"File archived without renaming: {archive_path}")
    except Exception as e:
        logging.error(f"Failed to archive file {file_path}: {str(e)}")

default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'depends_on_past': False,
    'max_active_tis_per_dag': OkeyConfig.MAX_CONCURRENT_TASKS,
}

with DAG(
    dag_id="okey_sales_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 7, 1),
    schedule_interval="0 11 * * *",  # каждый день в 15:00
    catchup=False,
    max_active_tasks=OkeyConfig.MAX_CONCURRENT_TASKS,
    concurrency=OkeyConfig.MAX_CONCURRENT_TASKS,
    tags=["okey"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    scanned_files = scan_okey_files()

    with TaskGroup("okey_file_processing_group") as processing_group:
        process_okey_file.expand(file_path=scanned_files)

    start >> scanned_files >> processing_group >> end

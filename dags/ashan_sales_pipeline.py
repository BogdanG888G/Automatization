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
from typing import List, Optional
from airflow.utils.task_group import TaskGroup
import chardet

# Database connection strings
DEFAULT_CONN_TEST = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)
DEFAULT_CONN_STAGE = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)

class Config:
    MAX_CONCURRENT_TASKS = 2
    MAX_FILES_PER_RUN = 5
    TASK_TIMEOUT = timedelta(minutes=30)
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

    DATA_DIR = "/opt/airflow/data"
    ARCHIVE_DIR = "/opt/airflow/archive"
    ALLOWED_EXT = {'.csv', '.xlsx', '.xls', '.xlsb'}
    FILE_PREFIX = "aushan_"
    STAGE_SCHEMA = "ashan"

    @staticmethod
    def get_processing_pool():
        try:
            pools = Variable.get("airflow_pools", default_var="default_pool")
            return "ashan_file_pool" if "ashan_file_pool" in pools.split(",") else "default_pool"
        except:
            return "default_pool"

    PROCESSING_POOL = get_processing_pool()
    POOL_SLOTS = 4

    CONN_SETTINGS = {
        'pool_size': 1,
        'max_overflow': 2,
        'pool_timeout': 30,
        'pool_recycle': 3600,
        'connect_args': {
            'timeout': 600,
            'application_name': 'airflow_ashan_loader'
        }
    }

_engine_cache = {}
_last_cache_update = datetime.min

def get_engine(db_type: str):
    """Get SQLAlchemy engine with caching"""
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

            _engine_cache[db_type] = create_engine(conn_str, **Config.CONN_SETTINGS)
        except Exception as e:
            logging.error(f"Error creating engine for {db_type}: {str(e)}")
            raise

    return _engine_cache[db_type]

def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding using chardet"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read first 10KB to detect encoding
        result = chardet.detect(raw_data)
        return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'

def preprocess_large_csv(file_path: str) -> str:
    """Pre-process large CSV files that might have parsing issues"""
    temp_path = file_path + ".processed"
    encoding = detect_file_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding) as infile, \
             open(temp_path, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                # Basic cleaning
                line = line.replace('\x00', '').strip()
                if not line:
                    continue
                    
                # Ensure proper line structure
                if line.count(';') >= 5:  # Minimum expected columns
                    outfile.write(line + '\n')
                elif i == 0:  # Header row
                    outfile.write(line + '\n')
                
        return temp_path
    except Exception as e:
        logging.error(f"File preprocessing error: {str(e)}")
        return file_path  # fallback to original

def read_csv_with_fallback(file_path: str) -> Optional[pd.DataFrame]:
    """Read CSV with multiple fallback strategies"""
    encoding = detect_file_encoding(file_path)
    
    # First try standard read
    try:
        return pd.read_csv(file_path, delimiter=';', decimal=',', thousands=' ', encoding=encoding)
    except Exception as e:
        logging.warning(f"Standard read failed, trying fallbacks: {str(e)}")
    
    # Try different strategies
    strategies = [
        {'engine': 'python', 'encoding': encoding},
        {'error_bad_lines': False, 'warn_bad_lines': True},
        {'encoding': 'cp1251'},
        {'encoding': 'latin1'},
        {'sep': ';', 'decimal': ',', 'thousands': ' ', 'engine': 'python'}
    ]
    
    for strategy in strategies:
        try:
            return pd.read_csv(file_path, **strategy)
        except:
            continue
    
    # Final attempt with preprocessing
    processed_path = preprocess_large_csv(file_path)
    try:
        return pd.read_csv(processed_path, delimiter=';', decimal=',', thousands=' ')
    finally:
        if processed_path != file_path and os.path.exists(processed_path):
            os.remove(processed_path)
    
    return None

@task(
    task_id="scan_files",
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=Config.TASK_TIMEOUT,
    pool=Config.PROCESSING_POOL,
    pool_slots=1
)
def scan_files() -> List[str]:
    """Scan for files matching criteria"""
    try:
        valid_files = []
        total_size = 0

        for root, _, files in os.walk(Config.DATA_DIR):
            for f in files:
                if (f.lower().startswith(Config.FILE_PREFIX) and
                        os.path.splitext(f)[1].lower() in Config.ALLOWED_EXT):

                    file_path = os.path.join(root, f)
                    file_size = os.path.getsize(file_path)

                    if total_size + file_size > Config.MAX_FILE_SIZE:
                        logging.warning(f"File size limit exceeded. Skipping {file_path}")
                        continue

                    valid_files.append(file_path)
                    total_size += file_size

                    if len(valid_files) >= Config.MAX_FILES_PER_RUN:
                        break

            if len(valid_files) >= Config.MAX_FILES_PER_RUN:
                break

        logging.info(f"Found {len(valid_files)} files (total size: {total_size / 1024 / 1024:.2f} MB)")
        return valid_files[:Config.MAX_FILES_PER_RUN]

    except Exception as e:
        logging.error(f"File scanning error: {str(e)}", exc_info=True)
        raise

@task(
    task_id="process_file",
    retries=2,
    retry_delay=timedelta(minutes=5),
    execution_timeout=Config.TASK_TIMEOUT,
    pool=Config.PROCESSING_POOL,
    pool_slots=1
)
def process_file(file_path: str):
    """Process a single file"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        elif file_path.lower().endswith(('.xlsx', '.xls', '.xlsb')):
            from common.convert_xlsx_to_csv import convert_excel_to_csv
            csv_paths = convert_excel_to_csv(file_path, max_rows=10000)
            logging.info(f"Converted Excel to CSV files: {csv_paths}")
            # Если несколько csv, можно брать первый (или расширить логику)
            file_path = csv_paths[0]
            df = read_csv_with_fallback(file_path)
        else:
            df = read_csv_with_fallback(file_path)

        if df is None or df.empty:
            raise ValueError("Failed to read file data")

        # Database operations
        engine_test = get_engine('test')
        try:
            from ashan.create_table_and_upload import create_table_and_upload
            table_name = create_table_and_upload(file_path, engine=engine_test)
            logging.info(f"Data loaded to raw.{table_name}")
        except exc.SQLAlchemyError as e:
            engine_test.dispose()
            raise RuntimeError(f"Raw load error: {str(e)}")

        engine_stage = get_engine('stage')
        try:
            from ashan.convert_raw_to_stage import convert_raw_to_stage
            convert_raw_to_stage(
                table_name=table_name,
                raw_engine=engine_test,
                stage_engine=engine_stage,
                stage_schema=Config.STAGE_SCHEMA,
                limit=10000
            )
            logging.info("Data loaded to stage")
        except exc.SQLAlchemyError as e:
            engine_stage.dispose()
            raise RuntimeError(f"Stage load error: {str(e)}")
        finally:
            engine_test.dispose()
            engine_stage.dispose()

        # Archive file
        os.makedirs(Config.ARCHIVE_DIR, exist_ok=True)
        archive_path = os.path.join(Config.ARCHIVE_DIR, os.path.basename(file_path))
        shutil.move(file_path, archive_path)
        logging.info(f"File archived: {archive_path}")

    except Exception as e:
        logging.error(f"File processing error: {file_path} - {str(e)}", exc_info=True)
        raise

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'depends_on_past': False,
    'max_active_tis_per_dag': Config.MAX_CONCURRENT_TASKS,
}

with DAG(
    dag_id="ashan_sales_pipeline_optimized",
    default_args=default_args,
    start_date=datetime(2025, 7, 1),
    schedule_interval=None,
    catchup=False,
    max_active_tasks=Config.MAX_CONCURRENT_TASKS,
    concurrency=Config.MAX_CONCURRENT_TASKS,
    tags=["ashan", "optimized"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    scanned_files = scan_files()

    with TaskGroup("file_processing_group") as processing_group:
        process_file.expand(file_path=scanned_files)

    start >> scanned_files >> processing_group >> end
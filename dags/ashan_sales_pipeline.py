from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import os
import shutil
import logging
from sqlalchemy import create_engine, exc
from typing import List
from airflow.utils.task_group import TaskGroup

# Твои строки подключения:
DEFAULT_CONN_TEST = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)
DEFAULT_CONN_STAGE = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Stage"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)

# Конфигурация
class Config:
    MAX_CONCURRENT_TASKS = 4
    MAX_FILES_PER_RUN = 20
    TASK_TIMEOUT = timedelta(minutes=30)

    DATA_DIR = "/opt/airflow/data"
    ARCHIVE_DIR = "/opt/airflow/archive"
    ALLOWED_EXT = {'.csv', '.xlsx', '.xls', '.xlsb'}
    FILE_PREFIX = "aushan_"
    STAGE_SCHEMA = "ashan"

    # Пул
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
        'pool_size': 5,
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
    """Получаем движок с кешированием и строкой подключения из констант"""
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
                raise ValueError(f"Неизвестный тип БД для подключения: {db_type}")

            _engine_cache[db_type] = create_engine(conn_str, **Config.CONN_SETTINGS)
        except Exception as e:
            logging.error(f"Ошибка создания engine для {db_type}: {str(e)}")
            raise

    return _engine_cache[db_type]


@task(
    task_id="scan_files",
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=Config.TASK_TIMEOUT,
    pool=Config.PROCESSING_POOL,
    pool_slots=1
)
def scan_files() -> List[str]:
    try:
        valid_files = []
        total_size = 0
        max_size = 2 * 1024 * 1024 * 1024  # 2GB

        for root, _, files in os.walk(Config.DATA_DIR):
            for f in files:
                if (f.lower().startswith(Config.FILE_PREFIX) and
                        os.path.splitext(f)[1].lower() in Config.ALLOWED_EXT):

                    file_path = os.path.join(root, f)
                    file_size = os.path.getsize(file_path)

                    if total_size + file_size > max_size:
                        logging.warning(f"Превышен лимит размера файлов (2GB). Пропускаем {file_path}")
                        continue

                    valid_files.append(file_path)
                    total_size += file_size

                    if len(valid_files) >= Config.MAX_FILES_PER_RUN:
                        break

            if len(valid_files) >= Config.MAX_FILES_PER_RUN:
                break

        logging.info(f"Найдено {len(valid_files)} файлов (общий размер: {total_size / 1024 / 1024:.2f} MB)")
        return valid_files[:Config.MAX_FILES_PER_RUN]

    except Exception as e:
        logging.error(f"Ошибка сканирования файлов: {str(e)}", exc_info=True)
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
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        if file_path.lower().endswith(('.xlsx', '.xls', '.xlsb')):
            from common.convert_xlsx_to_csv import convert_excel_to_csv
            file_path = convert_excel_to_csv(file_path)
            logging.info(f"Конвертирован в CSV: {file_path}")

        engine_test = get_engine('test')
        try:
            from ashan.create_table_and_upload import create_table_and_upload
            table_name = create_table_and_upload(file_path, engine=engine_test)
            logging.info(f"Данные загружены в raw.{table_name}")
        except exc.SQLAlchemyError as e:
            engine_test.dispose()
            raise RuntimeError(f"Ошибка загрузки в raw: {str(e)}")

        engine_stage = get_engine('stage')
        try:
            from ashan.convert_raw_to_stage import convert_raw_to_stage
            convert_raw_to_stage(
                table_name=table_name,
                raw_engine=engine_test,
                stage_engine=engine_stage,
                stage_schema=Config.STAGE_SCHEMA,
                limit=10000  # Добавлен параметр limit с значением 10 000
            )
            logging.info(f"Данные загружены в stage")
        except exc.SQLAlchemyError as e:
            engine_stage.dispose()
            raise RuntimeError(f"Ошибка загрузки в stage: {str(e)}")
        finally:
            engine_test.dispose()
            engine_stage.dispose()

        os.makedirs(Config.ARCHIVE_DIR, exist_ok=True)
        archive_path = os.path.join(Config.ARCHIVE_DIR, os.path.basename(file_path))
        shutil.move(file_path, archive_path)
        logging.info(f"Файл перемещен в архив: {archive_path}")

    except Exception as e:
        logging.error(f"Ошибка обработки файла {file_path}: {str(e)}", exc_info=True)
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

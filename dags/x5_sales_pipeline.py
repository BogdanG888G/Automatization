from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime
import os
import shutil
import logging
from sqlalchemy import create_engine, text

from utils import get_files_from_directory
from convert_xlsx_to_csv import convert_xlsx_to_csv
from create_table_and_upload import create_table_and_upload
from call_stored_procedure import call_stored_procedure

logger = logging.getLogger(__name__)

DATA_DIR = "/opt/airflow/data"
ARCHIVE_DIR = "/opt/airflow/archive"

# Установите значение по умолчанию для переменной
DEFAULT_CONN_STR = "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"

def get_engine():
    """Создаёт SQLAlchemy engine для MSSQL"""
    try:
        conn_str = Variable.get("MSSQL_CONN_STR")
    except KeyError:
        logger.warning("Using default connection string")
        conn_str = DEFAULT_CONN_STR
    return create_engine(conn_str)

@task
def scan_files():
    files = get_files_from_directory(DATA_DIR)
    logger.info(f"Найдено файлов для обработки: {files}")
    return files

@task
def process_file(file_path: str):
    try:
        logger.info(f"Начинаем обработку файла: {file_path}")

        # Конвертация если xlsx
        if file_path.endswith(".xlsx"):
            file_path = convert_xlsx_to_csv(file_path)

        # Создаём и загружаем данные
        engine = get_engine()
        table_name = create_table_and_upload(file_path, engine=engine)

        # Вызываем хранимую процедуру
        network = os.path.basename(file_path).split("_")[0].lower()
        call_stored_procedure(network, engine=engine)

        # Архивируем файл
        if not os.path.exists(ARCHIVE_DIR):
            os.makedirs(ARCHIVE_DIR)
        shutil.move(file_path, os.path.join(ARCHIVE_DIR, os.path.basename(file_path)))

        logger.info(f"Файл успешно обработан и перемещён в архив: {file_path}")

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file_path}: {e}")
        raise

with DAG(
    dag_id="x5_sales_pipeline",
    start_date=datetime(2025, 7, 7),
    schedule_interval="@daily",
    catchup=False,
    tags=["x5", "sales"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    files = scan_files()
    processed = process_file.expand(file_path=files)

    start >> files >> processed >> end
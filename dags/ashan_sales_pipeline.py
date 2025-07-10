import sys
sys.path.append('/opt/airflow/scripts')  # Убедись, что это путь до common/

from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from datetime import datetime
import os
import shutil
import logging
from sqlalchemy import create_engine

from common.utils import get_files_from_directory
from common.convert_xlsx_to_csv import convert_excel_to_csv  # ← исправлено имя модуля
from common.call_stored_procedure import call_stored_procedure
from ashan.create_table_and_upload import create_table_and_upload

logger = logging.getLogger(__name__)

DATA_DIR = "/opt/airflow/data"
ARCHIVE_DIR = "/opt/airflow/archive"

DEFAULT_CONN_STR = (
    "mssql+pyodbc://airflow_agent:123@host.docker.internal/Test"
    "?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)

def get_engine():
    try:
        conn_str = Variable.get("MSSQL_CONN_STR")
    except KeyError:
        logger.warning("Using default connection string")
        conn_str = DEFAULT_CONN_STR
    return create_engine(conn_str)

@task
def scan_files():
    files = get_files_from_directory(DATA_DIR)

    allowed_ext = ['.csv', '.xlsx', '.xls', '.xlsb']
    filtered = [
        f for f in files
        if os.path.basename(f).lower().startswith("ashan")
        and os.path.splitext(f)[1].lower() in allowed_ext
    ]
    logger.info(f"Ашан-файлы для обработки: {filtered}")
    return filtered

@task
def process_file(file_path: str):
    try:
        logger.info(f"Начинаем обработку файла Ашан: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.xlsx', '.xls', '.xlsb']:
            file_path = convert_excel_to_csv(file_path)

        engine = get_engine()
        create_table_and_upload(file_path, engine=engine)
        call_stored_procedure("ashan", engine=engine)

        if not os.path.exists(ARCHIVE_DIR):
            os.makedirs(ARCHIVE_DIR)
        shutil.move(file_path, os.path.join(ARCHIVE_DIR, os.path.basename(file_path)))

        logger.info(f"Файл Ашан успешно обработан и архивирован: {file_path}")

    except Exception as e:
        logger.error(f"Ошибка при обработке файла Ашан {file_path}: {e}")
        raise

with DAG(
    dag_id="ashan_sales_pipeline",
    start_date=datetime(2025, 7, 7),
    schedule_interval="@daily",
    catchup=False,
    tags=["ashan", "sales"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    files = scan_files()
    processed = process_file.expand(file_path=files)

    start >> files >> processed >> end
